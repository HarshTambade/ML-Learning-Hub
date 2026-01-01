"""Decision Tree Regression - Predicting Continuous Values

Demonstrates:
- Building regression trees
- MSE loss instead of classification
- Predicting house prices
- Residual analysis
- Performance metrics for regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

print("="*70)
print("DECISION TREE REGRESSION")
print("="*70)

# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# Create regression tree
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_regressor.fit(X_train, y_train)

# Predictions
y_pred_train = dt_regressor.predict(X_train)
y_pred_test = dt_regressor.predict(X_test)

# Metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nRegression Metrics:")
print(f"  Train MSE: {mse_train:.4f}")
print(f"  Test MSE: {mse_test:.4f}")
print(f"  Test RMSE: {rmse_test:.4f}")
print(f"  Test MAE: {mae_test:.4f}")
print(f"  Test R²: {r2_test:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Tree
plot_tree(dt_regressor, feature_names=feature_names, filled=True, ax=axes[0, 0], fontsize=8)
axes[0, 0].set_title("Regression Tree Structure", fontweight='bold')

# Predictions vs Actual
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.6)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title('Predictions vs Actual', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred_test
axes[1, 0].scatter(y_pred_test, residuals, alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Feature Importance
feature_imp = dt_regressor.feature_importances_
axes[1, 1].barh(feature_names, feature_imp, color='steelblue')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Feature Importance', fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('regression_tree.png', dpi=100, bbox_inches='tight')
print("\nRegression tree visualization saved as 'regression_tree.png'")
plt.show()

print("\n" + "="*70)
print("REGRESSION EXAMPLE COMPLETE")
print("="*70)
