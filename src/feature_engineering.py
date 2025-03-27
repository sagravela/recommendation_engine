import pandas as pd

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
