#!/usr/bin/env python3
"""
Regularization Techniques: Ridge and Lasso Regression

This script demonstrates:
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (combination of L1 and L2)
- Hyperparameter tuning with cross-validation
- Feature selection using regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Dataset prepared and scaled")
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print()

# ============================================================================
# 2. TRAIN MODELS WITH DIFFERENT REGULARIZATION TECHNIQUES
# ============================================================================

# Baseline model (no regularization)
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
baseline_r2 = r2_score(y_test, y_pred_baseline)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)

print("Baseline Model (No Regularization):")
print(f"  R² Score: {baseline_r2:.4f}")
print(f"  MSE: {baseline_mse:.4f}")
print()

# Ridge Regression (L2 regularization)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)

print("Ridge Regression (L2 Regularization, alpha=1.0):")
print(f"  R² Score: {ridge_r2:.4f}")
print(f"  MSE: {ridge_mse:.4f}")
print()

# Lasso Regression (L1 regularization)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)

print("Lasso Regression (L1 Regularization, alpha=0.1):")
print(f"  R² Score: {lasso_r2:.4f}")
print(f"  MSE: {lasso_mse:.4f}")
print(f"  Number of non-zero coefficients: {np.sum(lasso_model.coef_ != 0)}")
print()

# Elastic Net (L1 + L2 regularization)
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
y_pred_elasticnet = elasticnet_model.predict(X_test)
elasticnet_r2 = r2_score(y_test, y_pred_elasticnet)
elasticnet_mse = mean_squared_error(y_test, y_pred_elasticnet)

print("Elastic Net (L1 + L2 Regularization, alpha=0.1, l1_ratio=0.5):")
print(f"  R² Score: {elasticnet_r2:.4f}")
print(f"  MSE: {elasticnet_mse:.4f}")
print()

# ============================================================================
# 3. HYPERPARAMETER TUNING - FINDING BEST ALPHA
# ============================================================================
alphas = np.logspace(-4, 2, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))
    
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

best_ridge_alpha = alphas[np.argmax(ridge_scores)]
best_lasso_alpha = alphas[np.argmax(lasso_scores)]

print("Hyperparameter Tuning Results:")
print(f"  Best Ridge alpha: {best_ridge_alpha:.6f} (R² = {max(ridge_scores):.4f})")
print(f"  Best Lasso alpha: {best_lasso_alpha:.6f} (R² = {max(lasso_scores):.4f})")
print()

# ============================================================================
# 4. FEATURE COEFFICIENTS ANALYSIS
# ============================================================================
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Baseline': baseline_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_,
    'ElasticNet': elasticnet_model.coef_
})

print("Feature Coefficients:")
print(coeff_df)
print()

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model Performance Comparison
ax1 = axes[0, 0]
models = ['Baseline', 'Ridge', 'Lasso', 'ElasticNet']
r2_scores = [baseline_r2, ridge_r2, lasso_r2, elasticnet_r2]
colors = ['blue', 'green', 'red', 'orange']
ax1.bar(models, r2_scores, color=colors, alpha=0.7)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim([0, 1])
for i, v in enumerate(r2_scores):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Alpha vs R2 for Ridge and Lasso
ax2 = axes[0, 1]
ax2.plot(alphas, ridge_scores, 'o-', label='Ridge', linewidth=2)
ax2.plot(alphas, lasso_scores, 's-', label='Lasso', linewidth=2)
ax2.axvline(best_ridge_alpha, color='green', linestyle='--', alpha=0.5, label=f'Best Ridge')
ax2.axvline(best_lasso_alpha, color='red', linestyle='--', alpha=0.5, label=f'Best Lasso')
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (Regularization Strength)')
ax2.set_ylabel('R² Score')
ax2.set_title('Hyperparameter Tuning: Alpha vs R²')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Coefficients Comparison
ax3 = axes[1, 0]
x_pos = np.arange(len(feature_names))
width = 0.2
ax3.bar(x_pos - width*1.5, baseline_model.coef_, width, label='Baseline', alpha=0.8)
ax3.bar(x_pos - width/2, ridge_model.coef_, width, label='Ridge', alpha=0.8)
ax3.bar(x_pos + width/2, lasso_model.coef_, width, label='Lasso', alpha=0.8)
ax3.bar(x_pos + width*1.5, elasticnet_model.coef_, width, label='ElasticNet', alpha=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(feature_names, rotation=45, ha='right')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Feature Coefficients Across Models')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Actual vs Predicted for Best Models
ax4 = axes[1, 1]
ax4.scatter(y_test, y_pred_baseline, alpha=0.5, label='Baseline', s=50)
ax4.scatter(y_test, y_pred_ridge, alpha=0.5, label='Ridge', s=50)
ax4.scatter(y_test, y_pred_lasso, alpha=0.5, label='Lasso', s=50)
min_val = min(y_test.min(), y_pred_baseline.min())
max_val = max(y_test.max(), y_pred_baseline.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')
ax4.set_xlabel('Actual Values')
ax4.set_ylabel('Predicted Values')
ax4.set_title('Actual vs Predicted')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'regularization_analysis.png'")
plt.show()

# ============================================================================
# 6. KEY INSIGHTS
# ============================================================================
print("\n=== Key Insights ===")
print("\nRegularization Techniques:")
print("1. Ridge (L2): Shrinks coefficients, keeps all features")
print("2. Lasso (L1): Shrinks coefficients, performs feature selection")
print("3. ElasticNet: Combines benefits of Ridge and Lasso")
print(f"\nLasso Feature Selection: {np.sum(lasso_model.coef_ != 0)} out of {len(feature_names)} features selected")
print(f"\nBest Ridge performs better when: All features are relevant")
print(f"Best Lasso performs better when: Only some features are relevant")
print(f"\nHyperparameter selection is crucial for regularization effectiveness")
print(f"Use cross-validation to find optimal alpha values")
