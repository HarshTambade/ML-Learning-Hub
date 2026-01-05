"""Model training utilities.

Provides utilities for training ML models.
"""

import numpy as np
from typing import Callable


class Trainer:
    """Base trainer for ML models."""
    
    def __init__(self, model, loss_fn: Callable, optimizer: str = 'sgd'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'loss': []}
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True):
        """Train the model."""
        for epoch in range(epochs):
            self.model.fit(X, y)
            predictions = self.model.predict(X)
            loss = self.loss_fn(y, predictions)
            self.history['loss'].append(loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return self.history
    
    def get_history(self) -> dict:
        """Get training history."""
        return self.history


class EarlyStoppingCallback:
    """Early stopping callback for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
    
    def __call__(self, loss: float) -> bool:
        """Check if should stop training.
        
        Returns:
            True if training should stop, False otherwise
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class LearningRateScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, initial_lr: float, schedule: str = 'constant'):
        self.initial_lr = initial_lr
        self.schedule = schedule
        self.epoch = 0
    
    def get_lr(self) -> float:
        """Get learning rate for current epoch."""
        if self.schedule == 'constant':
            return self.initial_lr
        elif self.schedule == 'exponential':
            return self.initial_lr * (0.95 ** self.epoch)
        elif self.schedule == 'linear':
            return self.initial_lr * (1 - self.epoch / 100)
        else:
            return self.initial_lr
    
    def step(self):
        """Move to next epoch."""
        self.epoch += 1
