from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np

from .utils import log

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

# Function to create cumulative lists
def create_cumulative_list(items: pd.Series) -> list:
    cumulative_list = []
    result = []
    for item in items:
        result.append(cumulative_list.copy())
        cumulative_list.append(item)
    
    return result

# Function to create sequences
def create_feature_sequence(tbl: pd.DataFrame, feature: str, fix_len: int = 5) -> pd.DataFrame:
    """
    Create sequential features related to the **last 5 products the user clicked on**, given by `product_id` and their corresponding categories, given by `category_name`.
    The current dataset has insufficient clicks per user to establish a reliable sequential feature, as only slightly more than 20% of users have more than one click. This limitation is primarily due to the data covering only a single week of user interactions. Nonetheless, I will proceed with creating sequential features, as these are expected to become more valuable when more extensive interaction data becomes available in the future.
    """
    name = f"seq_{feature}"
    # Sort values by user_id and time and create list of value for each user
    tbl[name] = tbl.sort_values(by=['user_id', 'time']).groupby('user_id')[feature].transform(create_cumulative_list)
    # Pad sequences with zeros
    tbl[name] = tbl[name].apply(lambda x: (x + [0] * fix_len)[:fix_len])
    # Cast to string
    tbl[name] = tbl[name].apply(lambda x: [str(p) for p in x])

    return tbl

def create_dataset(data_df: pd.DataFrame, features: list) -> tf.data.Dataset:
    # Convert pandas dataframe to tensorflow dataset
    data_dict = {}
    # Sequential features has to be handled differently
    for f in features:
        f = f.split('-')[1] if '-' in f else f
        data_dict[f] = tf.constant(data_df[f].to_list(), dtype=tf.string) if f.startswith('seq_') else data_df[f]
    return tf.data.Dataset.from_tensor_slices(data_dict)

if __name__=="__main__":
    processed_path = Path("data") / "processed"
    # Load data
    data_path = processed_path / "search_sample_data.parquet"
    data_df = pd.read_parquet(data_path)
    log.info("Data loaded: /%s", data_path)
    # Get products dataframe
    products_df = get_products_df(data_df)
    # Get clicks dataset
    clicks_df = get_clicks_df(data_df, products_df)
    # Create sequential features
    clicks_df = create_feature_sequence(clicks_df, 'product_id')
    clicks_df = create_feature_sequence(clicks_df, 'category_name')
    log.info("Dataframes processed.")

    # Save dataframes
    products_df_path = processed_path / "products_df.parquet"
    clicks_df_path = processed_path / "clicks_df.parquet"
    products_df.to_parquet(products_df_path)
    clicks_df.to_parquet(clicks_df_path)
    log.info("Dataframes saved to /%s and /%s", products_df_path, clicks_df_path)
