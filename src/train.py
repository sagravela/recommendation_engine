import os
import json

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from prepare_data_model import TRAIN_SIZE, create_dataset
from model import RecommenderEngineModel
from callbacks import CustomEarlyStopping

# Train Function
# Training Function
def train_model(
        model: RecommenderEngineModel,
        train: tf.data.Dataset, 
        val: tf.data.Dataset = None,
        params: dict = None,
        logging: bool = False,
        profile: tuple | int = 0,
        verbose: int = 1
    ) -> RecommenderEngineModel:
    """
    Trains a custom recommendation model using TensorFlow's training framework. The function configures the model based on given parameters, 
    and optionally enables logging and profiling.

    Parameters
    ----------
    train : tf.data.Dataset
        The training dataset.
    val : tf.data.Dataset, optional
        The validation dataset. Default is None.
    preprocessing : bool, optional
        Whether to apply preprocessing to the input datasets. Default is False.
    params : dict, optional
        Dictionary containing model parameters, including early stopping, learning rate schedule, and batch size.
    train_metrics : bool, optional
        Whether to enable training metrics. Default is False.
    logging : bool, optional
        If True, TensorBoard logging is enabled, and model parameters are saved to a log directory. Default is True.
    profile : tuple or int, optional
        Batch to profile in TensorBoard for performance tracking. Default is 0 (no profiling).
    verbose : int, optional
        Verbosity level for model training. Default is 1.

    Returns
    -------
    model : RecommenderEngineModel
        The trained recommendation model instance.
    """

    # Setup Early Stopping and TerminateOnNaN callbacks
    callbacks = [CustomEarlyStopping(**params['EARLY_STOPPING']), tf.keras.callbacks.TerminateOnNaN()]
    
    # Setup learning rate scheduler with exponential decay
    train_samples = train # TODO
    n_steps = np.ceil(train_samples / params.get("BATCH_SIZE", 1024))
    params['LEARNING_RATE']['decay_steps'] = n_steps # decay after each epoch
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(**params['LEARNING_RATE'])    

    # Optionally enable TensorBoard logging
    if logging:
        logdir = params.get("LOGDIR", "logs")
        # Create a file writer for the log directory
        file_writer = tf.summary.create_file_writer(logdir)
        
        # Write model parameters to log directory
        model_metadata = json.dumps(params, indent=4)
        with file_writer.as_default():
            tf.summary.text(f"Parameters for {logdir}:", f"```\n{model_metadata}\n```", step=0)

        # Add TensorBoard callback for logging and profiling
        callbacks.append(tf.keras.callbacks.TensorBoard(logdir, profile_batch=profile))

    # Choose and configure the optimizer
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    if params["MODEL"]["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile the model
    model.compile(optimizer=optimizer)
    
    # Batch, cache, and prefetch the datasets for optimized loading during training
    train = train.cache().repeat().batch(params.get("BATCH_SIZE", 1024)).prefetch(tf.data.AUTOTUNE)
    if val:
        val = val.cache().repeat().batch(512).prefetch(tf.data.AUTOTUNE)

    val_samples = (1 - TRAIN_SIZE) * train_samples / TRAIN_SIZE # Amount of validation rows
    # Fit the model to the training data, with validation on the validation set
    model.fit(
        train,
        epochs=params.get("MAX_EPOCHS", 10),
        validation_data=val,
        callbacks=callbacks,
        steps_per_epoch=n_steps,
        validation_steps=np.ceil((val_samples / 512)),
        verbose=verbose
    )

    return model
## KFold Cross Validation function
def train_CV(params: dict, df: pd.DataFrame, user_features: list[str], product_features: list[str]):
    base_path = params['LOGDIR']
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    for fold, (train_index, val_index) in tqdm(enumerate(kf.split(df)), total=n_splits, desc='Cross Validation', unit='fold'):
        params['LOGDIR'] = os.path.join(base_path, f"fold_{fold}")
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]

        # Convert to TF Dataset
        train_ds = create_dataset(train_df, user_features + product_features)
        val_ds = create_dataset(val_df, user_features + product_features)

        # Train the model
        model = train_model(
            train= train_ds,
            val= val_ds,
            preprocessing= True, 
            params= params,
            logging= True,
            verbose= 0
        )

        # Get metrics
        result = {m: v[-1] for m, v in model.history.history.items() if m.startswith('val')}
        result['fold'] = fold
        result['n_epochs'] = len(model.history.history['loss'])
        results.append(result)

    return pd.DataFrame(results)