from pathlib import Path
from itertools import zip_longest

import tensorflow as tf
import pandas as pd
import numpy as np

from src.train_functions import train_model
from src.utils import log

class FeatureSelection():
    def __init__(self, train: tf.data.Dataset, val: tf.data.Dataset, params: dict, threshold: float = 0.05):
        self.train = train
        self.val = val
        self.params = params
        self.path = params["logs_path"]
        self.threshold = threshold
        # Reset the model features
        self.params['tower']['query'] = ["time", "score", "cat-user_id"]
        self.params['tower']['candidate'] = ["cat-product_id"]
        
        # Remove the initial user and product features
        self.model_user_features = [e for e in params['tower']['query'] if e not in self.params['tower']['query']]
        self.model_product_features = [e for e in params['tower']['candidate'] if e not in self.params['tower']['candidate']]

        # Array to save metrics for each model
        self.results = []

        # Set the order of features
        self.features = list(zip_longest(self.model_user_features, self.model_product_features))
        
        self.retrieval_metric_value = -np.inf
        self.rating_metric_value = np.inf    
    
    def _train_model(self):
        """
        Add feature to the model, train it and return metrics.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        dict
            History of the model
        """
        # Save logging
        model = train_model(
            train= self.train,
            val= self.val,
            preprocessing= False,
            params= self.params,
            logging= True,
            verbose= 0
        )
        result = {k: v[-1] for k, v in model.history.history.items() if k.startswith('val')}
        result['selected'] = False
        current_retrieval_value = result['val_factorized_top_k/top_50_categorical_accuracy']
        current_rating_value = result['val_root_mean_squared_error']

        # Keep the feature only if it improves one of the metrics and if both metrics don't exceed the threshold
        if (current_retrieval_value >= self.retrieval_metric_value or \
            current_rating_value <= self.rating_metric_value) and \
            (current_retrieval_value > self.retrieval_metric_value - self.threshold and \
            current_rating_value < self.rating_metric_value + self.threshold):
            self.retrieval_metric_value = current_retrieval_value
            self.rating_metric_value = current_rating_value
            result['selected'] = True # set flag to true if the feature will be added to the model
        return result

    def _add_feature(self, tower: str, feature: str):
        self.params["tower"][tower.upper()].append(feature)
        self.params["logs_path"] = self.path / f"add_{tower}_{feature}"
        log.info(f"Added {feature} feature to the {tower.capitalize()} Tower. Fitting...")
        result = self._train_model()
        if not result['selected']:
            log.info(f"Feature named {feature} to {tower} tower doesn't improve the model.")
            self.params['tower'][tower.upper()].remove(feature) # Remove feature from the parameters
        return {'tower': tower, 'feature': feature, **result}
    
    def run(self) -> None:
        # Step by step, I will select one feature from user followed by one from product until all features are selected
        # Run the initial baseline model with only two features without adding any feature
        log.info(f"Training baseline model")
        self.params["logs_path"] = self.path / f"baseline"
        self.results.append({'tower': '', 'feature': 'baseline', **self.train()})

        for user_feature, product_feature in self.features:
            # Add a feature to the user, train the model and save result
            if user_feature:
                self.results.append(self._add_feature('query', user_feature))

            # Add a feature to the product, train the model and save result
            if product_feature:
                self.results.append(self._add_feature('candidate', product_feature))
        
        # Save results for further analysis
        pd.DataFrame(self.results).to_csv(self.path / "results.csv", index=False)
        return
    