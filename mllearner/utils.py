"""Utility functions for ML workflows."""

import numpy as np
import pickle
from typing import Any


class ModelSaver:
    """Save and load ML models."""
    
    @staticmethod
    def save_model(model: Any, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class DataValidator:
    """Validate data for ML tasks."""
    
    @staticmethod
    def check_nan_values(data: np.ndarray) -> bool:
        """Check if data contains NaN values."""
        return np.isnan(data).any()
    
    @staticmethod
    def check_shape(X: np.ndarray, y: np.ndarray) -> bool:
        """Check if X and y have compatible shapes."""
        return X.shape[0] == y.shape[0]


class HyperparameterTuner:
    """Hyperparameter tuning utilities."""
    
    def __init__(self, model, param_grid: dict):
        self.model = model
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = -float('inf')
    
    def grid_search(self, X: np.ndarray, y: np.ndarray, scoring_fn):
        """Perform grid search over parameter grid."""
        def generate_combinations(grid, keys, idx=0, current={}):
            if idx == len(keys):
                yield current.copy()
            else:
                key = keys[idx]
                for val in grid[key]:
                    current[key] = val
                    yield from generate_combinations(grid, keys, idx + 1, current)
        
        for params in generate_combinations(self.param_grid, list(self.param_grid.keys())):
            for key, val in params.items():
                setattr(self.model, key, val)
            
            self.model.fit(X, y)
            score = scoring_fn(self.model, X, y)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        return self.best_params
