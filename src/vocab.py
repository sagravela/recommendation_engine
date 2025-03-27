
import tensorflow as tf
import pandas as pd
import numpy as np

def set_vocabulary(ds: tf.data.Dataset, cat_features: list) -> dict:
    """
        Create the vocabulary for each feature. `hour` and `day_of_week` are time related features that will be created from `time` feature.
    """
    # Define vocabulary for categorical features
    vocabulary = {}
    # Add hour and day of the week vocabulary manually, since they are not in the dataset
    vocabulary['hour'] = np.arange(0, 24, dtype=np.int32)
    vocabulary['day_of_week'] = np.arange(0, 7, dtype=np.int32)

    for feature in cat_features:
        vocab = ds.batch(512).map(lambda x: x[feature])
        vocabulary[feature] = np.unique(np.concatenate(list(vocab)))
    return vocabulary

def feature_range(df: pd.DataFrame, column: str) -> tuple[float, float]:
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
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    # Ensure that the lower bound is equal or over the miminum value allowed
    return (lower_bound if lower_bound >= df[column].min() else df[column].min(), upper_bound)

def set_buckets(df: pd.DataFrame, features: list[str]) -> dict:
    # Get continuous features boundaries metadata
    buckets = {}
    for feature in features:
        boundaries = feature_range(df, feature)
        buckets[feature] = np.linspace(boundaries[0], boundaries[1], num=100)
    return buckets
