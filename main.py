

from src.prepare_data_model import prepare_data, split_data


# `time` won't be used in the model directly, but it will be used to calculate related time features
# `score` is the target variable to predict in ranking model
USER_FEATURES = [
    'time',         
    'cat-user_id',
    'cat-channel',
    'cat-device_type',
    'text-query_text',
    'seq-seq_product_id',
    'seq-seq_category_name',
    'score',
    ]

PRODUCT_FEATURES = [
    'cat-product_id',
    'cat-category_name',
    'cat-merchant_name',
    'cat-merchant_city',
    'cat-merchant_state',
    'cat-merchant_region',
    'int-free_shipping',
    'int-is_sold_out',
    'int-editor_pick',
    'int-on_sale',
    'text-product_name',
    'cont-sales_last_week', 
    'cont-sales_last_month', 
    'cont-sales_last_year',            
    'cont-price_in_cents',
    'cont-reviews',
]

def main():
    