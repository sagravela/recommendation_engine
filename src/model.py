import re

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


## Embedding Layer
@tf.keras.saving.register_keras_serializable(package="RecommendationEngine", name="EmbeddingsLayer")
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, name: str, params: dict, feature_dim: dict):
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
        self.feature_dim = feature_dim

        l_name = name.replace("Embeddings", "").lower()
        self.features = params['tower'][l_name.upper()]
        self.emb_weight = params['model'].get('emb_weight', 1)

        self.embeddings = {}
        self.output_dim = 0

        for feature in self.features:
            if '-' not in feature:
                continue

            prep, feat = feature.split("-")
            if prep in ["cat", "int"]:
                emb_dim, embedding = self.get_embedding(feature)
                self.embeddings[feature] = embedding

            elif prep == "text":
                emb_dim, embedding = self.get_embedding(feature)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GlobalAveragePooling1D(name= f"text_avg_{feature}"),
                ])

            elif prep == "disc":
                emb_dim, embedding = self.get_embedding(feature)
                self.embeddings[f"disc-clip_{feat}"] = embedding

            elif prep == "seq":
                emb_dim, embedding = self.get_embedding(feature)
                self.embeddings[feature] = tf.keras.Sequential([
                    embedding,
                    tf.keras.layers.GRU(emb_dim, name=f"seq_gru_{feature}"),
                ])

    def get_embedding(self, feature: str) -> tuple[int, tf.keras.layers.Layer]:
        """
        Generate an embedding layer for the given feature

        Parameters
        ----------
        feature : str
            Name of the feature
        feature_dim : int
            Input dimension of the embedding layer

        Returns
        -------
        tf.keras.layers.Layer
            Embedding layer
        """
        feature_dim = self.feature_dim[feature]
        emb_dim = int((np.log2(feature_dim) + 1) * self.emb_weight)
        self.output_dim += emb_dim
        return emb_dim, tf.keras.layers.Embedding(feature_dim + 1, emb_dim, name=f"emb_{feature}", mask_zero=feature.split("-")[0] == "text")

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
        deep_layers = model_params[f"{l_name}_layers"]
        model = tf.keras.Sequential(name=f"{l_name}_deep_layers")

        if model_params["cross_layer"]:
            # Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost.
            # In practice, it've been observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
            model.add(tfrs.layers.dcn.Cross(projection_dim=input_dim // 4, kernel_initializer="glorot_uniform", name=f"cross_{l_name}"))

        # Use the ReLU activation for all but the last layer.
        for i, layer_size in enumerate(deep_layers[:-1]):
            model.add(tf.keras.layers.Dense(layer_size, activation="relu", name=f"{l_name}_layer{i:02d}"))

            # Add dropout layer for regularization after each layer except between the last two layers
            if i != len(self.deep_layers) - 1:
                model.add(tf.keras.layers.Dropout(model_params["dropout"], name=f"{l_name}_dropout{i:02d}"))

        # No activation for the output layer.
        model.add(tf.keras.layers.Dense(deep_layers[-1], name=f"{l_name}_output_layer"))

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
            candidates: tf.data.Dataset,
            feature_dim: dict,
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
        candidates : tf.data.Dataset
            Processed candidates dataset.
        train_metrics : bool, optional
            Whether to enable training metrics. Default is False
        """
        super().__init__()

        self.params = params
        self.train_metrics = train_metrics

        # Input Deep Layers verification
        assert params['model']['user_layers'][-1] == params['model']['product_layers'][-1], "User and Product output layers must have the same dimension"
        assert params['model']['rating_layers'][-1] == 1, "Rating output layer must have 1 unit"

        # QUERY
        self.user_embedding: tf.Tensor = Embeddings('UserEmbeddings', params, feature_dim)
        self.user_model = DeepLayers('UserTower', params['model'], self.user_embedding.output_dim)

        # PRODUCT
        self.product_embedding: tf.Tensor = Embeddings('ProductEmbeddings', params, feature_dim)
        self.product_model = DeepLayers('ProductTower', params['model'], self.product_embedding.output_dim)

        # RATING
        self.rating_model: tf.Tensor = DeepLayers(
            'RatingModel',
            params['model'],
            self.user_embedding.output_dim + self.product_embedding.output_dim
        )

        # TASKS
        # Retrieval Task
        # Build the candidates model to be used as candidates.
        self.candidates_model = tf.keras.Sequential(name="product_candidates")
        self.candidates_model.add(self.product_embedding)
        self.candidates_model.add(self.product_model)

        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                # candidates here is the preprocessed version of products
                candidates.batch(512).cache().prefetch(tf.data.AUTOTUNE).map(self.candidates_model, num_parallel_calls=tf.data.AUTOTUNE)
            )
        )

        # Ranking Task
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, input: dict[str, tf.Tensor]):
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
            'train_metrics': self.train_metrics
        })
        return config
