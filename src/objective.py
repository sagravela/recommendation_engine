import os
import argparse

import tensorflow as tf
import optuna

from src.train import train_model

# Hyperparameter Tuning
## Objective Function
# Objective function to be maximized.
def objective(trial, params: dict, train: tf.data.Dataset, val: tf.data.Dataset):
    """
    Optuna Objective Function to be optimized.
    """    
    def deep_layers(trial, units: list) -> list:
        """
        Helper function to suggest deep layers arquitectures.

        Parameters
        ----------
        trial : optuna.trial
            Optuna trial
        units : list
            List of units
        Returns
        -------
        list
            Deep layers arquitectures
        """
        # I will ensure in rating model at least one deep layer
        deep_layers = trial.suggest_int("rating_layers", 1, 2)

        layers = []
        for l in range(deep_layers):            
            layers.append(trial.suggest_categorical(f'rating_units_l{l}', units))
        # Add 1 unit in output layer by regression
        layers.append(1)
        return layers
    

    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()
    
    # Define parameters and suggest values to be tuned and optimized
    params['LOGDIR'] = os.path.join("output", "logs", "optuna", "run_{:02d}".format(trial.number))
    params['LEARNING_RATE']['initial_learning_rate'] = trial.suggest_categorical('initial_learning_rate', [0.1, 0.01])
    params['MODEL']['emb_weight'] = trial.suggest_int('emb_weight', 4, 16, step=4)
    output_layer = trial.suggest_categorical('output_layer', [8, 16, 32, 64])
    params['MODEL']['user_layers'] = [output_layer]
    params['MODEL']['product_layers'] = [output_layer]
    params['MODEL']['rating_layers'] = deep_layers(trial, [8, 16, 32, 64, 128, 256])
    params['MODEL']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    params['MODEL']['cross_layer'] = trial.suggest_categorical('cross_layer', [True, False])
    params['MODEL']['optimizer'] = trial.suggest_categorical('optimizer', ['Adagrad', 'Adam'])

    model = train_model(
        train= train,
        val = val,
        preprocessing= False, 
        params= params,
        logging= True,
        verbose= 0
    )
    
    # Get metrics
    trial_results = {m: v[-1] for m, v in model.history.history.items() if m.startswith('val')}
      
    # Save metrics as attributes for further analysis
    for m, v in trial_results.items():
        trial.set_user_attr(m, v)
    
    # Return the objective values
    return (
        trial_results['val_factorized_top_k/top_50_categorical_accuracy'],
        trial_results['val_root_mean_squared_error']
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="TODO"
    )
    parser.add_argument(
        "-n", "--n_trials", type=int, default = 100, help="Number of trials."
    )
    args = parser.parse_args()

    # Load storage
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage("hp.log"),
    )

    # Create a study
    study = optuna.create_study(
        storage=storage,
        study_name="recommendation_engine",
        directions=['maximize', 'minimize'],
        load_if_exists=True
    )

    # Perform optimization
    study.optimize(objective, n_trials= args.n_trials, show_progress_bar= True)
