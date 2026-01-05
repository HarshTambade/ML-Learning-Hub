"""Model evaluation metrics.

Provides various metrics for evaluating model performance.
"""

import numpy as np
from typing import Tuple


class ClassificationMetrics:
    """Metrics for classification models."""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision for binary classification."""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall for binary classification."""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        precision = ClassificationMetrics.precision(y_true, y_pred)
        recall = ClassificationMetrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        return np.array([[tn, fp], [fn, tp]])


class RegressionMetrics:
    """Metrics for regression models."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
