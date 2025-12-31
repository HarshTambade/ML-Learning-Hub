"""Hyperparameter Tuning for Logistic Regression

This script demonstrates hyperparameter tuning using GridSearchCV
and RandomizedSearchCV with proper cross-validation.

Author: ML Learning Hub
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. CREATE DATASET
# ============================================================================

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dataset created")
print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Test samples: {X_test_scaled.shape[0]}")
print(f"Features: {X_train_scaled.shape[1]}")
print()

# ============================================================================
# 2. BASELINE MODEL
# ============================================================================

baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train_scaled, y_train)
baseline_score = baseline_model.score(X_test_scaled, y_test)

print(f"Baseline Model (default params) Accuracy: {baseline_score:.4f}")
print()

# ============================================================================
# 3. GRID SEARCH
# ============================================================================

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 500, 1000]
}

print("GridSearchCV - Searching optimal parameters...")
gridsearch = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

gridsearch.fit(X_train_scaled, y_train)

print(f"Best Parameters: {gridsearch.best_params_}")
print(f"Best CV Score: {gridsearch.best_score_:.4f}")
print(f"Test Score: {gridsearch.score(X_test_scaled, y_test):.4f}")
print()

# ============================================================================
# 4. RANDOMIZED SEARCH
# ============================================================================

print("RandomizedSearchCV - Faster search...")
random_search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    n_iter=20,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")
print(f"Test Score: {random_search.score(X_test_scaled, y_test):.4f}")
print()

# ============================================================================
# 5. COMPARISON
# ============================================================================

print("\nModel Comparison:")
print(f"Baseline Accuracy: {baseline_score:.4f}")
print(f"GridSearch Best Accuracy: {gridsearch.score(X_test_scaled, y_test):.4f}")
print(f"RandomSearch Best Accuracy: {random_search.score(X_test_scaled, y_test):.4f}")
print()

# ============================================================================
# 6. FEATURE IMPORTANCE FROM BEST MODEL
# ============================================================================

best_model = gridsearch.best_estimator_
print("Best Model Classification Report (Test Set):")
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print()

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

# Plot 1: GridSearch Results Summary
ax1 = axes[0]
results_df = pd.DataFrame(gridsearch.cv_results_)
results_df_sorted = results_df.sort_values('mean_test_score', ascending=False).head(10)
ax1.barh(range(len(results_df_sorted)), results_df_sorted['mean_test_score'])
ax1.set_xlabel('Mean CV Score')
ax1.set_title('Top 10 Parameter Combinations')
ax1.set_yticks(range(len(results_df_sorted)))
ax1.set_yticklabels([str(p)[:50] for p in results_df_sorted['params']], fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Model Performance Comparison
ax2 = axes[1]
models = ['Baseline', 'GridSearch', 'RandomSearch']
scores = [
    baseline_score,
    gridsearch.score(X_test_scaled, y_test),
    random_search.score(X_test_scaled, y_test)
]
colors = ['skyblue', 'lightcoral', 'lightgreen']
ax2.bar(models, scores, color=colors, alpha=0.7)
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Model Performance Comparison')
ax2.set_ylim([0.7, 1.0])
for i, v in enumerate(scores):
    ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Hyperparameter Tuning Complete!")
print("="*70)
