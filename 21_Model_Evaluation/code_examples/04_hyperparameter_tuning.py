"""Hyperparameter Tuning - Grid Search & Random Search
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
import numpy as np

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters (Grid): {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# Random Search
param_dist = {'C': np.logspace(-2, 2, 20), 'kernel': ['linear', 'rbf', 'poly']}
random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

print(f"\nBest parameters (Random): {random_search.best_params_}")
print(f"Test score: {random_search.score(X_test, y_test):.4f}")
