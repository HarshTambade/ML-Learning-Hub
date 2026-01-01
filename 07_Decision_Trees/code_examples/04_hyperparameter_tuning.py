"""
Hyperparameter Tuning for Decision Trees

This script demonstrates:
- Grid Search CV
- Randomized Search CV
- Hyperparameter tuning strategies
- Learning curves
- Model validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("HYPERPARAMETER TUNING FOR DECISION TREES")
print("=" * 80)

# Load data
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 1. GRID SEARCH CV
# ============================================================================

print("\n" + "=" * 80)
print("1. GRID SEARCH CV")
print("=" * 80)

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}

print(f"\nSearching over {len(param_grid['criterion']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations...")

# Grid Search with smaller grid for faster execution
small_param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    small_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Set Score: {grid_search.score(X_test, y_test):.4f}")

# Top 10 parameter combinations
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]

print(f"\nTop 10 Parameter Combinations:")
print("-" * 80)
for idx, row in top_10.iterrows():
    print(f"Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f}) | Params: {row['params']}")

# ============================================================================
# 2. RANDOMIZED SEARCH CV
# ============================================================================

print("\n" + "=" * 80)
print("2. RANDOMIZED SEARCH CV")
print("=" * 80)

param_dist = {
    'max_depth': list(range(1, 30)),
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_dist,
    n_iter=50,  # Number of iterations
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

random_search.fit(X_train, y_train)

print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")
print(f"Test Set Score: {random_search.score(X_test, y_test):.4f}")

# ============================================================================
# 3. IMPORTANT HYPERPARAMETERS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. INDIVIDUAL HYPERPARAMETER IMPACT")
print("=" * 80)

# Test max_depth impact
max_depths = list(range(1, 21))
depth_scores_train = []
depth_scores_test = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    depth_scores_train.append(dt.score(X_train, y_train))
    depth_scores_test.append(dt.score(X_test, y_test))

print(f"\nMax Depth Impact:")
for depth, train_score, test_score in zip(max_depths[:5], depth_scores_train[:5], depth_scores_test[:5]):
    print(f"Depth {depth:2d}: Train={train_score:.4f}, Test={test_score:.4f}")

# Test min_samples_split impact
min_samples = [2, 5, 10, 20, 50, 100, 200]
split_scores = []

for min_split in min_samples:
    dt = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
    dt.fit(X_train, y_train)
    split_scores.append(dt.score(X_test, y_test))

print(f"\nMin Samples Split Impact:")
for min_split, score in zip(min_samples, split_scores):
    print(f"Min Split {min_split:3d}: Test Score={score:.4f}")

# ============================================================================
# 4. LEARNING CURVES
# ============================================================================

print("\n" + "=" * 80)
print("4. LEARNING CURVES")
print("=" * 80)

best_dt = grid_search.best_estimator_

train_sizes, train_scores, val_scores = learning_curve(
    best_dt,
    X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    verbose=0
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print(f"\nLearning Curve Analysis:")
print(f"Training samples: {train_sizes}")
print(f"Final Training Score: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
print(f"Final Validation Score: {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")

# ============================================================================
# 5. METRICS EVALUATION OF BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("5. BEST MODEL PERFORMANCE METRICS")
print("=" * 80)

best_predictions = best_dt.predict(X_test)

accuracy = accuracy_score(y_test, best_predictions)
precision = precision_score(y_test, best_predictions)
recall = recall_score(y_test, best_predictions)
f1 = f1_score(y_test, best_predictions)

print(f"\nPerformance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, best_predictions))

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Max Depth Impact
ax1 = axes[0, 0]
ax1.plot(max_depths, depth_scores_train, label='Training Score', marker='o', linewidth=2)
ax1.plot(max_depths, depth_scores_test, label='Test Score', marker='s', linewidth=2)
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('Accuracy Score')
ax1.set_title('Impact of Max Depth on Model Performance')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Min Samples Split Impact
ax2 = axes[0, 1]
ax2.plot(min_samples, split_scores, marker='o', linewidth=2, color='green')
ax2.set_xlabel('Min Samples Split')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Impact of Min Samples Split on Model Performance')
ax2.grid(alpha=0.3)
ax2.set_xscale('log')

# 3. Learning Curves
ax3 = axes[1, 0]
ax3.plot(train_sizes, train_mean, label='Training Score', marker='o', linewidth=2)
ax3.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
ax3.plot(train_sizes, val_mean, label='Validation Score', marker='s', linewidth=2)
ax3.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
ax3.set_xlabel('Training Set Size')
ax3.set_ylabel('Accuracy Score')
ax3.set_title('Learning Curves - Best Model')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_ylim([0.8, 1.02])

# 4. Metrics Comparison
ax4 = axes[1, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ax4.bar(metrics, scores, color=colors)
ax4.set_ylabel('Score')
ax4.set_title('Best Model Performance Metrics')
ax4.set_ylim([0, 1.05])
for i, (metric, score) in enumerate(zip(metrics, scores)):
    ax4.text(i, score + 0.02, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("Hyperparameter Tuning Analysis Complete!")
print("=" * 80)
