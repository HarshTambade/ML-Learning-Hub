#!/usr/bin/env python3
"""
Advanced Topics in Linear Regression

This script demonstrates:
- Cross-validation techniques
- Gradient Descent optimization
- Normal Equation mathematical approach
- Model diagnostics and assumptions
- Practical best practices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# 1. CROSS-VALIDATION TECHNIQUES
# ============================================================================
print("=" * 70)
print("1. CROSS-VALIDATION TECHNIQUES")
print("=" * 70)

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')

print("\nK-Fold Cross-Validation (k=5):")
print(f"  Fold Scores: {kfold_scores}")
print(f"  Mean R²: {kfold_scores.mean():.4f} (+/- {kfold_scores.std() * 2:.4f})")

# Leave-One-Out Cross-Validation (computationally expensive)
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring='r2')
print(f"\nLeave-One-Out CV:")
print(f"  Mean R²: {loo_scores.mean():.4f}")
print(f"  Std Dev: {loo_scores.std():.4f}")

# Repeated K-Fold
repeat_kfold_scores = []
for i in range(3):
    kf = KFold(n_splits=5, shuffle=True, random_state=42+i)
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    repeat_kfold_scores.extend(scores)

print(f"\nRepeated K-Fold CV (3 repeats, k=5):")
print(f"  Mean R²: {np.mean(repeat_kfold_scores):.4f}")
print(f"  Std Dev: {np.std(repeat_kfold_scores):.4f}")

# ============================================================================
# 2. GRADIENT DESCENT VS NORMAL EQUATION
# ============================================================================
print("\n" + "=" * 70)
print("2. GRADIENT DESCENT VS NORMAL EQUATION")
print("=" * 70)

# Normal Equation approach (sklearn default)
model_normal = LinearRegression()
model_normal.fit(X_scaled, y)
y_pred_normal = model_normal.predict(X_scaled)
r2_normal = r2_score(y, y_pred_normal)

print(f"\nNormal Equation (Closed-form solution):")
print(f"  R² Score: {r2_normal:.4f}")
print(f"  MSE: {mean_squared_error(y, y_pred_normal):.4f}")
print(f"  Computation: O(n³) - Good for small to medium datasets")

# Gradient Descent approach (SGD)
model_sgd = SGDRegressor(loss='squared_error', learning_rate='invscaling', 
                         eta0=0.01, random_state=42, max_iter=1000)
model_sgd.fit(X_scaled, y)
y_pred_sgd = model_sgd.predict(X_scaled)
r2_sgd = r2_score(y, y_pred_sgd)

print(f"\nStochastic Gradient Descent:")
print(f"  R² Score: {r2_sgd:.4f}")
print(f"  MSE: {mean_squared_error(y, y_pred_sgd):.4f}")
print(f"  Computation: O(n*m) per iteration - Good for large datasets")

print(f"\nCoefficient Comparison:")
print(f"  Normal Equation coef (first 3): {model_normal.coef_[:3]}")
print(f"  SGD coef (first 3): {model_sgd.coef_[:3]}")

# ============================================================================
# 3. MODEL ASSUMPTIONS AND DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 70)
print("3. LINEAR REGRESSION ASSUMPTIONS")
print("=" * 70)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_train = LinearRegression()
model_train.fit(X_train, y_train)
y_pred_train = model_train.predict(X_train)
residuals = y_train - y_pred_train

print("\n1. Linearity: Check with residual plots")
print("2. Independence: Check autocorrelation (Durbin-Watson test)")

# Durbin-Watson statistic
dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"   Durbin-Watson Statistic: {dw_stat:.4f}")
print(f"   (Value close to 2 indicates no autocorrelation)")

print("\n3. Homoscedasticity: Check if variance is constant")
print(f"   Residuals mean: {np.mean(residuals):.6f} (should be ~0)")
print(f"   Residuals std: {np.std(residuals):.4f}")

print("\n4. Normality: Check if residuals are normally distributed")
_, p_value = stats.shapiro(residuals)
print(f"   Shapiro-Wilk test p-value: {p_value:.4f}")
print(f"   (p > 0.05 suggests normality)")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cross-validation comparison
ax1 = axes[0, 0]
fold_indices = np.arange(len(kfold_scores))
ax1.bar(fold_indices, kfold_scores, alpha=0.7, label='Fold Scores')
ax1.axhline(kfold_scores.mean(), color='r', linestyle='--', linewidth=2, label='Mean')
ax1.set_xlabel('Fold')
ax1.set_ylabel('R² Score')
ax1.set_title('K-Fold Cross-Validation Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals Distribution
ax2 = axes[0, 1]
ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax2.set_title('Residuals Distribution (should be Normal)')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Q-Q Plot (Normality Check)
ax3 = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Check for Normality)')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals vs Fitted Values
ax4 = axes[1, 1]
ax4.scatter(y_pred_train, residuals, alpha=0.6)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Fitted Values')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals vs Fitted Values (Check Homoscedasticity)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_topics_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'advanced_topics_analysis.png'")
plt.show()

# ============================================================================
# 5. PRACTICAL BEST PRACTICES
# ============================================================================
print("\n" + "=" * 70)
print("5. PRACTICAL BEST PRACTICES")
print("=" * 70)

print("""
1. Data Preprocessing:
   - Check for missing values
   - Scale/normalize features (important for gradient descent)
   - Remove outliers if necessary
   - Check for multicollinearity

2. Model Selection:
   - Always use cross-validation
   - Compare multiple models
   - Use appropriate evaluation metrics
   - Consider regularization for high-dimensional data

3. Model Interpretation:
   - Examine residuals for patterns
   - Check model assumptions
   - Look at feature importance/coefficients
   - Validate on unseen test data

4. Hyperparameter Tuning:
   - Use cross-validation for robust estimates
   - Use GridSearchCV or RandomizedSearchCV
   - Monitor both training and validation metrics

5. Production Deployment:
   - Version your model and data preprocessing
   - Monitor model performance over time
   - Retrain periodically with new data
   - Document assumptions and limitations
""")

print("\n" + "=" * 70)
print("Advanced Topics Completed!")
print("=" * 70)
