import os

import pandas as pd

def load_raw_data() -> pd.DataFrame:
    return pd.read_parquet(os.path.join("data", "search_sample_data.parquet"))

def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Rename and order columns
    columns = [
        'time', 'user_id', 'product_id', 'merchant_id', 'category_id', 
        'channel', 'query_text', 'device_type', 'sales_last_week',
        'sales_last_month', 'sales_last_year', 'previous_purchase',
        'free_shipping', 'is_sold_out', 'editor_pick', 'merchant_name',
        'product_name', 'price_in_cents', 'on_sale', 'category_name',
        'merchant_city', 'merchant_state', 'merchant_region', 'reviews',
        'add_to_cart', 'click', 'conversion'
    ]
    raw_data.columns = raw_data.columns.str.lower()
    data_df = (
        raw_data
        .rename(columns = {'visitor_token':'user_id'}, inplace = True)
        .reindex(columns, axis = 1)
    )

    return data_df

def preprocess_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # Convert to numeric
    cols_to_int = [
        'previous_purchase', 'free_shipping', 'is_sold_out', 'editor_pick',
        'on_sale', 'price_in_cents', 'reviews', 'add_to_cart', 'click', 'conversion'
    ]
    data_df[cols_to_int] = data_df[cols_to_int].astype(int)

    # Order by timestamp, neeeded by sequential features
    return data_df.sort_values(by=['time']).reset_index(drop = True)

def prepare_data() -> pd.DataFrame:
    raw_data = load_raw_data()
    cleaned_df = clean_data(raw_data)
    processed_df = preprocess_data(cleaned_df)
    return processed_df
