import os

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Train size
TRAIN_SIZE = 0.8

def get_products_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    ## Items Dataset
    For the items dataset, I will retain only the most recent features (ordered by time) to represent the current state of each product. This is because certain features, such as `free_shipping`, `price_in_cents`, `reviews`, etc., fluctuate over time.
    The primary keys for the items dataset are `product_id` and `merchant_id`. However, since `merchant_id` can be mapped to a specific `merchant_name`, I will use `merchant_name` as the merchant identifier in the model.
    Due to some inconsistencies in spelling, I will ensure that each [`product_id`, `merchant_name`] group has a single [`product_name`, `category_name`]. To achieve this, I will select the most frequent value (the mode) for this group to serve as the correct one.
    In a real scenario, I would have a item database that would be up-to-date with the latest features for each product.
    """
    products_df = df.groupby(["product_id", "merchant_name"]).last().reset_index() # Retain only last product

    # Items dataset doesn't need time feature anymore
    return products_df.drop("time", axis=1)

def get_clicks_df(
        df: pd.DataFrame,
        products_df: pd.DataFrame,
        add_to_cart_score: float= 0.5, 
        conversion_score: float = 1.0
    ) -> pd.DataFrame:
    """
    ## Clicks Dataset
    In order to do a retrieval task, I need to separate positive interactions from negative ones. **Clicks are assumed as positive interactions**, so those interactions without any click are considered as negative.
    In the other hand, I need a **score feature** to rank products by the feedback received by the user. I will use the following weights to create this feature:

    |Score|Add to Cart|Conversion|
    |----|----|----|
    |`0.0`|No|No|
    |`0.5`|Yes|No|
    |`1.0`|No/Yes|Yes|
    """
    # Create score feature
    df = df[df['click'] == True].reset_index(drop=True)
    df['score'] = np.where(df['conversion'] == True, conversion_score, np.where(df['add_to_cart'] == True, add_to_cart_score, 0.0))
    
    # Convert `time` to string given that TensorFlow does not support `datetime64` data type.
    df['time'] = df['time'].astype(str)

    # modify clicks dataset with the new values of product_name and category_name
    merged = df.merge(products_df, on = ['product_id', 'merchant_name'])
    df[['product_name', 'category_name']] = merged[['product_name_y', 'category_name_y']]

    return df

def create_dataset(data_df: pd.DataFrame, features: list) -> tf.data.Dataset:
    # Convert pandas dataframe to tensorflow dataset
    data_dict = {}
    # Sequential features has to be handled differently
    for f in features:
        f = f.split('-')[1] if '-' in f else f
        data_dict[f] = tf.constant(data_df[f].to_list(), dtype=tf.string) if f.startswith('seq_') else data_df[f]
    return tf.data.Dataset.from_tensor_slices(data_dict)

def prepare_data(
        data_df: pd.DataFrame,
        user_features: list[str],
        product_features: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, tf.data.Dataset, tf.data.Dataset]:
    products_df = get_products_df(data_df)
    clicks_df = get_clicks_df(data_df, products_df)

    # Create datasets from the whole data to further vocabulary building
    clicks = create_dataset(clicks_df, user_features + product_features)
    products = create_dataset(products_df, product_features)

    # Save products dataset to 'data' folder for further use
    products.save(os.path.join('data','products'))
    return clicks_df, products_df, clicks, products

def split_data(
        clicks_df: pd.DataFrame,
        user_features: list[str],
        product_features: list[str]        
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    
    **NOTE**: I won't use *test set* because I want to have as many data as possible for training. I will use only 80% of data for training set and 20% for validation set.
    In a real world scenario, I should have a test set to evaluate the model but I would have more data as well.
    """
    # Data length
    n_data = clicks_df.shape[0]
    train_samples = int(TRAIN_SIZE * n_data)

    y = clicks_df['score']
    # Split without shuffling by the sequential features
    clicks_train_df, clicks_val_df = train_test_split(clicks_df, test_size=1-TRAIN_SIZE, random_state=42, shuffle=False)
    # Ensure clicks is not shuffled
    assert clicks_train_df['time'].is_monotonic_increasing and clicks_val_df['time'].is_monotonic_increasing, \
        "DataFrame is not ordered by timestamp in ascending order."
    # Split with shuffling in case the sequential features won't be used, stratify to ensure target consistency among sets
    clicks_train_df_sh, clicks_val_df_sh = train_test_split(clicks_df, test_size=1-TRAIN_SIZE, random_state=42, stratify=y)

    # Load as dataset
    clicks_train = create_dataset(clicks_train_df, user_features + product_features)
    clicks_val = create_dataset(clicks_val_df, user_features + product_features)
    clicks_train_sh = create_dataset(clicks_train_df_sh, user_features + product_features)
    clicks_val_sh = create_dataset(clicks_val_df_sh, user_features + product_features)

    return clicks_train, clicks_val, clicks_train_sh, clicks_val_sh