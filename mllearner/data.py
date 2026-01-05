"""Data handling and preprocessing module.

This module provides utilities for loading, exploring, and preprocessing data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union


class DataLoader:
    """Load and explore datasets."""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load CSV file into a DataFrame."""
        return pd.read_csv(filepath)
    
    @staticmethod
    def load_json(filepath: str) -> Union[pd.DataFrame, dict]:
        """Load JSON file."""
        return pd.read_json(filepath)
    
    @staticmethod
    def describe_data(data: pd.DataFrame) -> dict:
        """Get comprehensive data description."""
        return {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'statistics': data.describe().to_dict()
        }


class DataProcessor:
    """Process and transform data."""
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            method: 'mean', 'median', or 'drop'
        """
        if method == 'mean':
            return data.fillna(data.mean())
        elif method == 'median':
            return data.fillna(data.median())
        elif method == 'drop':
            return data.dropna()
        return data
    
    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        return data.drop_duplicates()
    
    @staticmethod
    def encode_categorical(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        return pd.get_dummies(data, columns=columns)
    
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """Standardize data (mean=0, std=1)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)


class TrainTestSplit:
    """Split data into training and testing sets."""
    
    @staticmethod
    def split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> Tuple:
        """Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
        """
        if random_state:
            np.random.seed(random_state)
        
        indices = np.random.permutation(len(X))
        test_idx = int(len(X) * test_size)
        
        train_idx = indices[test_idx:]
        test_indices = indices[:test_idx]
        
        return X[train_idx], X[test_indices], y[train_idx], y[test_indices]
