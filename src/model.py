import re

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

## Preprocessing Layer
@tf.function
def replace_empty_string(x: tf.Tensor) -> tf.Tensor:
    """
    Replace empty strings in a feature with '[NULL]'

    Parameters
    ----------
    x : tf.Tensor
        Feature to be processed

    Returns
    -------
    tf.Tensor
        Processed feature
    """
    return tf.where(tf.strings.regex_full_match(x, ""), tf.constant("[NULL]"), x)

@tf.function
def extract_features(x: dict[tf.Tensor]) -> dict[tf.Tensor]:
    """
    Extracts time-based features such as the hour and day of the week from a timestamp in the input using Zellers Congruence.

    Parameters
    ----------
    x : dict[tf.Tensor]
        Dictionary of input features

    Returns
    -------
    dict[tf.Tensor]
        Dictionary of processed features
    """
    # Extract the 'time' feature
    times = x['time']

    # Extract date and time parts using substr
    date_str = tf.strings.substr(times, 0, 10)
    time_str = tf.strings.substr(times, 11, 12)

    # Extract date components
    years = tf.strings.to_number(tf.strings.substr(date_str, 0, 4), tf.int32)
    months = tf.strings.to_number(tf.strings.substr(date_str, 5, 2), tf.int32)
    days = tf.strings.to_number(tf.strings.substr(date_str, 8, 2), tf.int32)

    # Extract hour
    hours = tf.strings.to_number(tf.strings.substr(time_str, 0, 2), tf.int32)

    # Helper function to calculate day of week using Zeller's Congruence
    def zellers_congruence(year, month, day):
        if month < 3:
            month += 12
            year -= 1
        K = year % 100
        J = year // 100
        f = day + ((13 * (month + 1)) // 5) + K + (K // 4) + (J // 4) + (5 * J)
        return (f % 7 + 5) % 7 # Shift the output to have Monday as 0, Tuesday as 1, ...

    # Calculate the day of the week
    day_of_week = tf.vectorized_map(lambda x: zellers_congruence(x[0], x[1], x[2]), (years, months, days))

    # Add the parsed components back to the dictionary
    return {**x, 'hour': hours, 'day_of_week': day_of_week}

@tf.function
def clip_cont_features(x: dict[tf.Tensor]) -> dict[tf.Tensor]:
    """
    Clip continuous features to their respective boundaries.

    Parameters
    ----------
    x : dict[tf.Tensor]
        Dictionary of input features
    features: list[str]
        List of features to be included in the model

    Returns
    -------
    dict[tf.Tensor]
        Dictionary of clipped features
    """
    clipped_features = {}
    for feature in x:
        # Clip features in BOUNDARIES and avoid redundancy when the same feature is mapped as norm and disc
        if feature in BOUNDARIES.keys() and f"clip_{feature}" not in clipped_features.keys():
            clipped_features[f"clip_{feature}"] = tf.clip_by_value(x[feature], BOUNDARIES[feature][0], BOUNDARIES[feature][1])
    # Add the generated clipped features to the input
    return {**x, **clipped_features}


@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="PreprocessingLayer")
class Preprocessing(tf.keras.layers.Layer):
  
    def __init__(self, name: str, features: list[str], ds: tf.data.Dataset= None):
        """
        A custom Keras preprocessing layer designed to handle various types of preprocessing 
        tasks such as categorical encoding, integer mapping, text vectorization, discretization, and 
        normalization.

        Parameters
        ----------
        name : str
            Name of the layer
        features : list[str]
            List of features to be processed
        ds : tf.data.Dataset, optional
            Dataset where the features come from (clicks or products), by default None
        """
        super().__init__(name=name)
        
        self.features = features
        self.prep_layers = {}
        # Save the vocabulary metadata for further use
        self.vocab = {}
        self.bucket = {}

        self.extract_time_features = None
        if "int-hour" in self.features or "int-day_of_week" in self.features:
            self.extract_time_features = tf.keras.layers.Lambda(extract_features, name= "extract_time_features")

        self.clip_outliers = None
        if any(f.startswith('disc-') or f.startswith('norm-') for f in self.features):
            self.clip_outliers = tf.keras.layers.Lambda(clip_cont_features, name= "clip_outliers", arguments={"features": self.features})

        for feature in self.features:
            if '-' not in feature:
                continue

            prep, feat = feature.split("-")
            if prep in ["cat", "seq"]:
                self.prep_layers[feature] = tf.keras.layers.StringLookup(name=feature).adapt(ds.map(lambda x: x[feat]))

            elif prep == "int":
                if feat == "hour":
                    self.prep_layers[feature] = tf.keras.layers.IntegerLookup(vocabulary = np.arange(0, 24, dtype=np.int32), name=feature)
                elif feat == "day_of_week":
                    self.prep_layers[feature] = tf.keras.layers.IntegerLookup(vocabulary = np.arange(0, 7, dtype=np.int32), name=feature)
                else:
                    self.prep_layers[feature] = tf.keras.layers.IntegerLookup(name=feature).adapt(ds.map(lambda x: x[feat]))
                
            elif prep == "text":
                text_layer = tf.keras.layers.TextVectorization(
                        max_tokens = 10_000,
                        output_mode="int",
                        output_sequence_length=20,
                        name=f"tv_{feature}"
                    ).adapt(ds.map(lambda x: x[feat]))
                self.prep_layers[feature] = tf.keras.Sequential([
                    tf.keras.layers.Lambda(replace_empty_string, name= f"text_null_{feature}"),
                    text_layer
                ], name=feature)

            elif prep == "disc":
                # Need to add clip before var name in order to use the clipped features rather than the original ones
                self.prep_layers[f"disc-clip_{feat}"] = tf.keras.layers.Discretization(num_bins = 100, name=feature).adapt(ds.map(lambda x: x[feat]))
            
            elif prep == "norm":
                self.prep_layers[f"norm-clip_{feat}"] = tf.keras.layers.Normalization(axis = None, name=feature).adapt(ds.map(lambda x: x[feat]))
            
            else:
                raise ValueError("Preprocessing type not supported.")

    def call(self, input: dict[str, tf.Tensor]):
        # Extract time features and add them if needed
        if self.extract_time_features:
            input = self.extract_time_features(input)
        # Clip outliers in continuous features if they are present
        if self.clip_outliers:
            input = self.clip_outliers(input)
        output = {feature: layer(input[feature.split("-")[1]]) for feature, layer in self.prep_layers.items()}
        return {**input, **output}
    
    def get_config(self):
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            'features': self.features,
            # Do not include 'ds' as it is not needed for serialization
        })
        return config

## Embedding Layer
@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="EmbeddingsLayer")
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, name: str, params: dict, prep_layer: Preprocessing):
        """
        A custom Keras layer designed to handle embedding generation for various types of features.
        The layer manages multiple embeddings and concatenates them to create a feature representation.

        Parameters
        ----------
        name : str
            Name of the layer
        params : dict[str, Any]
            Dictionary of layer parameters
        """
        super().__init__(name= name)

        self.params = params
        
        l_name = name.replace("Embeddings", "").lower()
        self.features = params['FEATURES'][l_name.upper()]
        self.emb_weight = params['MODEL'].get('emb_weight', 1)
        # Retrieve the vocabulary
        vocab, bucket = prep_layer.vocab, prep_layer.buckets
        
        self.embeddings = {}
        self.output_dim = 0

        for feature in self.features:
            if '-' not in feature:
                continue
            
            prep, feat = feature.split("-")
            if prep in ["cat", "int"]:
                input_dim = len(vocab[feat])
                emb_dim, embedding = self.get_embedding(feature, input_dim)
                self.embeddings[feature] = embedding
                
            elif prep == "text":
                input_dim = 10_000
                emb_dim, embedding = self.get_embedding(feature, input_dim)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GlobalAveragePooling1D(name= f"text_avg_{feature}"),
                ])
            
            elif prep == "disc":
                input_dim = len(bucket[feat])
                emb_dim, embedding = self.get_embedding(feature, input_dim)
                self.embeddings[f"disc-clip_{feat}"] = embedding

            elif prep == "seq":
                input_dim = len(vocab[feat.replace("seq_", "")])
                emb_dim, embedding = self.get_embedding(feature, input_dim)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GRU(emb_dim, name=f"seq_gru_{feature}"),
                ])
    
    def get_embedding(self, feature: str, input_dim: int) -> tuple[int, tf.keras.layers.Layer]:
        """
        Generate an embedding layer for the given feature

        Parameters
        ----------
        feature : str
            Name of the feature
        input_dim : int
            Input dimension of the embedding layer

        Returns
        -------
        int
            Embedding dimension
        tf.keras.layers.Layer
            Embedding layer
        """
        emb_dim = int((np.log2(input_dim) + 1) * self.emb_weight)
        self.output_dim += emb_dim
        return emb_dim, tf.keras.layers.Embedding(input_dim + 1, emb_dim, name=f"emb_{feature}", mask_zero=feature.split("-")[0] == "text")

    def call(self, input: dict[str, tf.Tensor]):
        # Add normalized features as they are because they don't need embeddings. Only reshape is needed.
        normalized_features = [tf.reshape(input[f"norm-clip_{f.split('-')[1]}"], (-1, 1)) for f in self.features if f.split("-")[0] == "norm"]
        # Concat embeddings and normalized features
        return tf.concat([self.embeddings[feature](input[feature]) for feature in self.embeddings.keys()] + normalized_features, axis=1)

    def get_config(self):
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            'params': self.params
        })
        return config

## Deep Layers Model
@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="DeepLayers")
class DeepLayers(tf.keras.Model):
    def __init__(self, name: str, model_params: dict, input_dim: int):
        """
        A custom Keras model representing a deep neural network with optional cross-layer functionality. 
        This layer is designed to build a stack of dense layers with ReLU activation and include dropout for regularization.

        Parameters
        ----------
        name : str
            Name of the layer
        model_params : dict[str, Any]
            Dictionary of model parameters
        input_dim : int
            Input dimension
        """
        super().__init__(name= name)

        self.model_params = model_params
        l_name = re.sub(r"Tower|Model", "", name).lower()
        self.deep_layers = model_params[f"{l_name}_layers"]
        self.model = tf.keras.Sequential(name=f"{l_name}_deep_layers")

        if model_params["cross_layer"]:
            # Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost. 
            # In practice, it've been observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
            self.model.add(tfrs.layers.dcn.Cross(projection_dim=input_dim // 4, kernel_initializer="glorot_uniform", name=f"cross_{l_name}"))

        # Use the ReLU activation for all but the last layer.
        for i, layer_size in enumerate(self.deep_layers[:-1]):
            self.model.add(tf.keras.layers.Dense(layer_size, activation="relu", name=f"{l_name}_layer{i:02d}"))

            # Add dropout layer for regularization after each layer except between the last two layers
            if i != len(self.deep_layers) - 1:                
                self.model.add(tf.keras.layers.Dropout(model_params["dropout"], name=f"{l_name}_dropout{i:02d}"))

        # No activation for the output layer.
        self.model.add(tf.keras.layers.Dense(self.deep_layers[-1], name=f"{l_name}_output_layer"))
        
    def call(self, input: tf.Tensor):
        return self.model(input)
    
    def get_config(self):
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            'model_params': self.model_params
        })
        return config

## Recommender Engine Model
class RecommenderEngineModel(tfrs.models.Model):

    def __init__(
            self, 
            params: dict, 
            preprocessing: bool = None,
            prep_layers: tuple = None,
            clicks: tf.data.Dataset = None,
            prods: tf.data.Dataset = None,
            train_metrics: bool = False
        ):
        """
        A custom recommendation model built on top of TensorFlow Recommenders (TFRS) framework.
        This model implements a dual-tower architecture for learning user (query) and product (candidate) embeddings.
        It supports both retrieval (matching users to products) and ranking tasks (predicting user ratings for products). 
        Optionally, it handles input feature preprocessing, which is disabled during the experimentation workflow and enabled during serving.
        -  `prep_layer` only needed with preprocessing turned off.
        - `clicks` is only needed in preprocessing enabled.
        - `prods`is the raw candidates dataset in preprocessing = True, therefore is the preprocessed candidates dataset otherwise.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of model parameters
        preprocessing : bool, optional
            Whether to apply preprocessing. Default is False
        candidates : tf.data.Dataset
            Preprocessed Products dataset.
        train_metrics : bool, optional
            Whether to enable training metrics. Default is False
        """
        super().__init__()

        self.params = params
        self.train_metrics = train_metrics

        # If preprocessing is required
        self.preprocessing = preprocessing
        if self.preprocessing:
            self.user_prep = Preprocessing(
                "UserPreprocessing", params['FEATURES']['USER'], 
                clicks.batch(512).cache().prefetch(tf.data.AUTOTUNE)
            )
            self.prods_prep = Preprocessing(
                "ProductsPreprocessing", params['FEATURES']['PRODUCT'], 
                prods.batch(512).cache().prefetch(tf.data.AUTOTUNE)
            )
        else:
            self.user_prep, self.prods_prep = prep_layers

        # Input Deep Layers verification
        assert params['MODEL']['user_layers'][-1] == params['MODEL']['product_layers'][-1], "User and Product output layers must have the same dimension"
        assert params['MODEL']['rating_layers'][-1] == 1, "Rating output layer must have 1 unit"

        # QUERY
        self.user_embedding: tf.Tensor = Embeddings('UserEmbeddings', params, self.user_prep)
        self.user_model = DeepLayers('UserTower', params['MODEL'], self.user_embedding.output_dim)
        
        # PRODUCT
        self.product_embedding: tf.Tensor = Embeddings('ProductEmbeddings', params, self.prods_prep)
        self.product_model = DeepLayers('ProductTower', params['MODEL'], self.product_embedding.output_dim)
                
        # RATING
        self.rating_model: tf.Tensor = DeepLayers(
            'RatingModel', 
            params['MODEL'], 
            self.user_embedding.output_dim + self.product_embedding.output_dim
        )

        # TASKS
        # Retrieval Task
        # Build the candidates model to be used as candidates.
        self.candidates_model = tf.keras.Sequential(name="product_candidates")
        if self.preprocessing:
            self.candidates_model.add(self.prods_prep)
        self.candidates_model.add(self.product_embedding)
        self.candidates_model.add(self.product_model)

        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                # candidates here is the preprocessed version of products
                prods.batch(512).cache().prefetch(tf.data.AUTOTUNE).map(self.candidates_model, num_parallel_calls=tf.data.AUTOTUNE)
            )
        )

        # Ranking Task
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, input: dict[str, tf.Tensor]):
        # Make feature preprocessing if required
        if self.preprocessing:
            user_prep_data = self.user_prep(input)
            prods_prep_data = self.prods_prep(input)
            input = {**user_prep_data, **prods_prep_data}

        # Get embeddings
        user_embedding: tf.Tensor = self.user_embedding(input)
        product_embedding: tf.Tensor = self.product_embedding(input)

        # Get model outputs
        return (
            self.user_model(user_embedding),
            self.product_model(product_embedding),
            self.rating_model(tf.concat([user_embedding, product_embedding], axis=1))
        )

    def compute_loss(self, input: dict[str, tf.Tensor], training=False):
        ratings: tf.Tensor = input.pop("score")

        user, product, rating = self(input)

        compute_metrics = True if self.train_metrics else not training
        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating,
            compute_metrics=compute_metrics
        )
        retrieval_loss = self.retrieval_task(user, product, compute_metrics=compute_metrics)

        # And sum up the losses
        return (rating_loss + retrieval_loss)
    
    def get_config(self):
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            'params': self.params,
            'preprocessing': self.preprocessing
        })
        return config
