"""
Feature Importance in Decision Trees

This script demonstrates:
- Extracting feature importance from trained decision trees
- Visualizing feature contributions
- Comparing different importance measures
- Using feature importance for feature selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# 1. FEATURE IMPORTANCE BASICS
# ============================================================================

print("=" * 70)
print("FEATURE IMPORTANCE IN DECISION TREES")
print("=" * 70)

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train decision tree
dt_classifier = DecisionTreeClassifier(
    max_depth=4, random_state=42, criterion='gini'
)
dt_classifier.fit(X_train, y_train)

# Extract feature importance
feature_importance = dt_classifier.feature_importances_
feature_names = X.columns

print("\n1. Feature Importance (Gini-based):")
print("-" * 70)
for name, importance in zip(feature_names, feature_importance):
    print(f"{name:30s}: {importance:.4f} {'█' * int(importance * 100)}")

print(f"\nModel Accuracy: {dt_classifier.score(X_test, y_test):.4f}")

# ============================================================================
# 2. VISUALIZATION OF FEATURE IMPORTANCE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Horizontal Bar Chart
ax1 = axes[0, 0]
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

ax1.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
ax1.set_xlabel('Importance Score')
ax1.set_title('Feature Importance - Horizontal Bar Chart')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Vertical Bar Chart with color gradient
ax2 = axes[0, 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
ax2.bar(range(len(feature_importance)), feature_importance, color=colors)
ax2.set_xticks(range(len(feature_importance)))
ax2.set_xticklabels(feature_names, rotation=45, ha='right')
ax2.set_ylabel('Importance Score')
ax2.set_title('Feature Importance - Vertical Bar Chart')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Pie Chart
ax3 = axes[1, 0]
ax3.pie(feature_importance, labels=feature_names, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax3.set_title('Feature Importance Distribution')

# Plot 4: Cumulative Importance
ax4 = axes[1, 1]
sorted_idx = np.argsort(feature_importance)[::-1]
cumsum_importance = np.cumsum(feature_importance[sorted_idx])
ax4.plot(range(1, len(cumsum_importance) + 1), cumsum_importance, 'o-', 
         linewidth=2, markersize=8, color='darkred')
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Cumulative Importance')
ax4.set_xticks(range(1, len(feature_names) + 1))
ax4.set_title('Cumulative Feature Importance')
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('feature_importance_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n2. Feature Importance Visualization saved!")

# ============================================================================
# 3. FEATURE IMPORTANCE WITH BREAST CANCER DATASET (MORE FEATURES)
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE WITH BREAST CANCER DATASET")
print("=" * 70)

breast_cancer = load_breast_cancer()
X_cancer = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y_cancer = breast_cancer.target

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

dt_cancer = DecisionTreeClassifier(
    max_depth=5, random_state=42, criterion='gini'
)
dt_cancer.fit(X_train_bc, y_train_bc)

feature_importance_bc = dt_cancer.feature_importances_
feature_names_bc = X_cancer.columns

# Top 10 features
top_10_idx = np.argsort(feature_importance_bc)[-10:][::-1]

print("\n3. Top 10 Most Important Features (Breast Cancer Dataset):")
print("-" * 70)
for idx, i in enumerate(top_10_idx, 1):
    print(f"{idx:2d}. {feature_names_bc[i]:30s}: {feature_importance_bc[i]:.4f}")

print(f"\nModel Accuracy: {dt_cancer.score(X_test_bc, y_test_bc):.4f}")

# Visualize top 10 features
fig, ax = plt.subplots(figsize=(12, 8))
top_features_df = pd.DataFrame({
    'Feature': feature_names_bc[top_10_idx],
    'Importance': feature_importance_bc[top_10_idx]
})

colors_top = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features_df)))
ax.barh(top_features_df['Feature'], top_features_df['Importance'], color=colors_top)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top 10 Most Important Features - Breast Cancer Dataset', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('breast_cancer_top_10_features.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. REGRESSION TREE FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE IN REGRESSION TREES")
print("=" * 70)

# Create synthetic regression data
np.random.seed(42)
X_reg = np.random.randn(200, 5)
y_reg = (2 * X_reg[:, 0] - 3 * X_reg[:, 1] + 
         X_reg[:, 2] ** 2 + 0.5 * X_reg[:, 3] + np.random.normal(0, 0.1, 200))

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_regressor.fit(X_train_reg, y_train_reg)

feature_importance_reg = dt_regressor.feature_importances_
feature_names_reg = [f'Feature_{i}' for i in range(X_reg.shape[1])]

print("\n4. Feature Importance in Regression Tree:")
print("-" * 70)
for name, importance in zip(feature_names_reg, feature_importance_reg):
    print(f"{name}: {importance:.4f}")

print(f"\nR² Score: {dt_regressor.score(X_test_reg, y_test_reg):.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
colors_reg = plt.cm.coolwarm(np.linspace(0, 1, len(feature_importance_reg)))
ax.bar(feature_names_reg, feature_importance_reg, color=colors_reg)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance in Regression Tree', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('regression_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. IMPACT OF MAX_DEPTH ON FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("IMPACT OF MAX_DEPTH ON FEATURE IMPORTANCE")
print("=" * 70)

importances_by_depth = {}

for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='gini')
    dt.fit(X_train, y_train)
    importances_by_depth[depth] = dt.feature_importances_

print("\n5. Feature Importance Changes with Tree Depth:")
print("-" * 70)
for depth in range(1, 11):
    print(f"Max Depth {depth:2d}: {list(np.round(importances_by_depth[depth], 3))}")

# Visualize how importance changes with depth
fig, ax = plt.subplots(figsize=(14, 8))

for i, feature in enumerate(feature_names):
    importances = [importances_by_depth[depth][i] for depth in range(1, 11)]
    ax.plot(range(1, 11), importances, marker='o', label=feature, linewidth=2)

ax.set_xlabel('Max Depth', fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_title('How Feature Importance Changes with Tree Depth', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig('feature_importance_vs_depth.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. FEATURE IMPORTANCE FOR FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION USING IMPORTANCE THRESHOLDS")
print("=" * 70)

thresholds = [0.01, 0.05, 0.10, 0.15]

print("\n6. Feature Selection Results:")
print("-" * 70)

for threshold in thresholds:
    # Select features with importance >= threshold
    selected_features = feature_names[feature_importance >= threshold]
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Selected Features ({len(selected_features)}): {list(selected_features)}")
    
    # Train model with selected features only
    if len(selected_features) > 0:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        dt_selected = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_selected.fit(X_train_selected, y_train)
        accuracy = dt_selected.score(X_test_selected, y_test)
        print(f"Model Accuracy with selected features: {accuracy:.4f}")
    else:
        print("No features selected with this threshold")

print("\n" + "=" * 70)
print("Feature Importance Analysis Complete!")
print("=" * 70)
