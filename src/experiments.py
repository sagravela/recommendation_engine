from pathlib import Path
import time

import tensorflow as tf
import pandas as pd
import ImbalancedLearningRegression as iblr
import matplotlib.pyplot as plt

from src.data_processing import create_dataset
from src.train_functions import train_model, train_CV
from src.utils import log


def batch_size_exp(params: dict, batches: list[int]) -> None:
    """
    Function to perform batch size experimentation.
    """
    path = params["logs_path"]
    # I won't use sequential features for this analysis in order to apply CV
    params['tower']['query'].remove("seq-seq_product_id")
    params['tower']['query'].remove("seq-seq_category_name")

    # Try different batch sizes
    results = []
    for i, batch in enumerate(batches):
        params['model']['batch_size'] = batch
        params['logs_path'] = path / "bs_{}".format(batch)

        log.info(f"Training model for {batch} batch size ({i+1}/{len(batches)})")
        init_time = time.time()
        results_df = train_CV(params)
        end_time = time.time()
        results_df['batch_size'] = batch
        results_df['training_time'] = end_time - init_time
        results.append(results_df)

    pd.concat(results, ignore_index= True).to_csv(path / "results.csv", index=False)
    log.info("Results saved to /%s", path / "results.csv")
    return


def deep_layers_exp(params: dict, deep_layers: list) -> None:
    path = params["logs_path"]
    results = []
    for layers in deep_layers:
        # Use the number of layers as ID of each model
        n_layers = len(layers) - 1
        log.info(f"Training model with the following layers: \n- User Layers: {layers} \n- Product Layers: {layers}\n")
        params['logs_path'] = path / f'{n_layers}_deep_layers'
        # Update layers architecture for each tower
        params['model']['user_layers'] = layers
        params['model']['product_layers'] = layers

        result = train_CV(params)
        result['n_layers'] = n_layers
        results.append(result)

    pd.concat(results, ignore_index= True).to_csv(path / "results.csv", index="run_number")
    log.info("Results saved to /%s", path / "results.csv")
    return


def resample(data: pd.DataFrame, all_features: list[str]) -> tuple:
    # The rel_ctrl_pts_rg argument takes a 2d array (matrix).
    # It is used to manually specify the regions of interest or rare "minority" values in y.
    # The first column indicates the y values of interest, the second column indicates a mapped value of relevance, either 0 or 1,
    # where 0 is the least relevant and 1 is the most relevant, and the third column is indicative.
    # It will be adjusted afterwards, use 0 in most cases.
    rg_matrix = [
        [0.5, 1, 0], # minority class, high relevance
        [1.0, 1, 0], # minority class, high relevance
        [0, 0, 0] # majority class, low relevance
    ]

    # Random Oversample
    log.info("Random Oversample")
    ro_clicks_train_df = iblr.ro(
        data = data,
        y = 'score',
        rel_method="manual", # Set manual to use manual relevance control
        rel_ctrl_pts_rg= rg_matrix # Set relevance control points
    )

    # Random Undersampling
    log.info("Random Undersampling")
    ru_clicks_train_df = iblr.random_under(
        data = data,
        y = 'score',
        rel_method="manual",
        rel_ctrl_pts_rg= rg_matrix
    )

    # Gaussian Noise
    log.info("Gaussian Noise")
    gn_clicks_train_df = iblr.gn(
        data = data,
        y = 'score',
        rel_method="manual",
        rel_ctrl_pts_rg= rg_matrix
    )

    # Print shapes
    print(f"Original shape: {data.shape}")
    print(f"RO shape: {ro_clicks_train_df.shape}")
    print(f"RU shape: {ru_clicks_train_df.shape}")
    print(f"Gaussian Noise shape: {gn_clicks_train_df.shape}")

    # Load as dataset
    log.info("Load as dataset")
    features = [f for f in all_features if not f.startswith("seq")]
    orig_clicks_train = create_dataset(data, features)
    ro_clicks_train = create_dataset(ro_clicks_train_df, features)
    ru_clicks_train = create_dataset(ru_clicks_train_df, features)
    gn_clicks_train = create_dataset(gn_clicks_train_df, features)

    # Plot densities
    ro_clicks_train_df['score'].plot(kind="kde", label="Random Oversampling", title="Resampling Comparison")
    ru_clicks_train_df['score'].plot(kind="kde", label="Random Undersampling")
    gn_clicks_train_df['score'].plot(kind="kde", label="Gaussian Noise")
    data['score'].plot(kind="kde", label="Original")
    plt.xlabel("Score")
    plt.xticks([0, 0.5, 1])
    plt.legend()
    plt.show()

    return orig_clicks_train, ro_clicks_train, ru_clicks_train, gn_clicks_train


def resample_exp(params: dict, train_df: pd.DataFrame, val_ds: tf.data.Dataset, features: list[str]) -> None:
    path = params["logs_path"]
    log.info("Resample training")

    # Sequential features aren't accepted by the resampling methods.
    original_clicks_train_df = train_df.drop(['seq_product_id', 'seq_category_name'], axis=1).reset_index(drop=True)  # Has to be a pandas dataframe

    orig_clicks_train, ro_clicks_train, ru_clicks_train, gn_clicks_train = resample(original_clicks_train_df, features)
    # list of tf datasets
    train_sets = [orig_clicks_train, ro_clicks_train, ru_clicks_train, gn_clicks_train]
    train_names = ["original", "RO", "RU", "GN"]

    results = []
    for i, (train_ds, name) in enumerate(zip(train_sets, train_names)):
        params['logs_path'] = path / name
        log.info(f"Training model for {name} dataset ({i+1}/{len(train_sets)})")
        model = train_model(
            train = train_ds,
            val = val_ds,
            preprocessing= True,
            params= params,
            logging=True,
            verbose= 0
        )
        result = {k: v[-1] for k, v in model.history.history.items() if k.startswith('val')}
        results.append({'name': name, **result})

    return pd.DataFrame(results)
