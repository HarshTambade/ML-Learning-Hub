"""Model implementations for ML tasks.

This module provides various ML model implementations.
"""

import numpy as np
from typing import Tuple


class LinearRegression:
    """Simple Linear Regression model."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.iterations):
            predictions = np.dot(X, self.weights) + self.bias
            error = predictions - y
            
            self.weights -= self.learning_rate * (1/m) * np.dot(X.T, error)
            self.bias -= self.learning_rate * (1/m) * np.sum(error)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias


class LogisticRegression:
    """Logistic Regression for binary classification."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            error = predictions - y
            
            self.weights -= self.learning_rate * (1/m) * np.dot(X.T, error)
            self.bias -= self.learning_rate * (1/m) * np.sum(error)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(linear_pred) >= 0.5).astype(int)


class KNearestNeighbors:
    """K-Nearest Neighbors classifier."""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data."""
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            prediction = max(set(k_labels), key=k_labels.count)
            predictions.append(prediction)
        return np.array(predictions)


class DecisionNode:
    """Node in decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """Simple Decision Tree classifier."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None
    
    def gini(self, y):
        """Calculate Gini impurity."""
        counter = {}
        for val in y:
            counter[val] = counter.get(val, 0) + 1
        impurity = 1
        for val in counter.values():
            prob = val / len(y)
            impurity -= prob ** 2
        return impurity
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build decision tree."""
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        """Recursively build tree."""
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) == 0:
            return DecisionNode(value=max(set(y), key=list(y).count) if len(y) > 0 else 0)
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gini = (np.sum(left_mask) * self.gini(y[left_mask]) + 
                        np.sum(right_mask) * self.gini(y[right_mask])) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        if best_feature is None:
            return DecisionNode(value=max(set(y), key=list(y).count))
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: DecisionNode):
        """Traverse tree to make prediction."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
