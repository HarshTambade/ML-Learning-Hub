"""Visualization utilities for ML.

Provides visualization tools for data and model insights.
"""

import numpy as np
from typing import List


class PlotUtils:
    """Utility functions for plotting."""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray):
        """Plot confusion matrix."""
        print("Confusion Matrix:")
        print(cm)
    
    @staticmethod
    def plot_training_history(history: dict):
        """Plot training history."""
        if 'loss' in history:
            losses = history['loss']
            print(f"Training Loss History (epochs: {len(losses)})")
            print(f"Initial Loss: {losses[0]:.4f}")
            print(f"Final Loss: {losses[-1]:.4f}")
    
    @staticmethod
    def plot_feature_importance(features: List[str], importance: List[float]):
        """Plot feature importance."""
        sorted_indices = np.argsort(importance)[::-1]
        print("Feature Importance:")
        for idx in sorted_indices:
            print(f"  {features[idx]}: {importance[idx]:.4f}")
    
    @staticmethod
    def print_metrics(metrics: dict):
        """Print evaluation metrics."""
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
