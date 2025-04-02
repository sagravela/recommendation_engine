import tensorflow as tf
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
def feature_range(x: tf.Tensor) -> tuple[float, float]:
    """
    The distribution of continuous features present skewness and outliers.

    To reduce the impact of outliers on the model, I decide to dentify and clip outliers in continuous features by the following equation:

    $$
    Q_1 - 3 \cdot IQR \leq x \leq Q_3 + 3 \cdot IQR
    $$
    Where `Q1` and `Q3` are the first and third quartiles respectively, and `IQR` is the interquartile range given by:

    $$
    IQR = Q3 - Q1
    $$

    I will get the allowed boundaries for each continuous feature in `BOUNDARIES` dictionary. Then, I will clip the outliers within the model.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.quantile(x, 0.25)
    Q3 = np.quantile(x, 0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    # Ensure that the lower bound is equal or over the miminum value allowed
    return (lower_bound if lower_bound >= np.min(x) else np.min(x), upper_bound)

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
    features = set([f for f in x.keys() if f.startswith('disc-') or f.startswith('norm-')])
    for feature in features:
        boundarie = feature_range(x[feature])
        clipped_features[f"clip_{feature}"] = tf.clip_by_value(x[feature], boundarie[0], boundarie[1])
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

        self.extract_time_features = None
        if "int-hour" in self.features or "int-day_of_week" in self.features:
            self.extract_time_features = tf.keras.layers.Lambda(extract_features, name= "extract_time_features")

        self.clip_outliers = None
        if any(f.startswith('disc-') or f.startswith('norm-') for f in self.features):
            self.clip_outliers = tf.keras.layers.Lambda(clip_cont_features, name= "clip_outliers")

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
        return {**input, **output} # Output features will replace input features

    def get_config(self):
        # Method needed for serialization and saving
        config = super().get_config()
        config.update({
            'features': self.features,
            # Do not include 'ds' as it is not needed for serialization
        })
        return config
