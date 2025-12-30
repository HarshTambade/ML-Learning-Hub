#!/usr/bin/env python3
"""
Multiple Linear Regression Implementation and Demonstration

This script demonstrates:
- How Multiple Linear Regression works with multiple features
- Feature importance and relationships
- Model evaluation with multiple features
- Handling multicollinearity
- Feature scaling and normalization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target shape: {y.shape}")
print()

# Create a DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("First few rows:")
print(df.head())
print()

print("Statistical summary:")
print(df.describe())
print()

# ============================================================================
# 2. SPLIT DATA
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print()

# ============================================================================
# 3. FEATURE SCALING (Important for MLR)
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled to have mean=0 and std=1")
print(f"Scaled training data shape: {X_train_scaled.shape}")
print()

# ============================================================================
# 4. CREATE AND TRAIN THE MODEL
# ============================================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully")
print(f"Intercept: {model.intercept_:.4f}")
print()

print("Feature Coefficients:")
for feature, coef in zip(feature_names, model.coef_):
    print(f"  {feature:>20}: {coef:>8.4f}")
print()

# ============================================================================
# 5. ANALYZE FEATURE IMPORTANCE
# ============================================================================
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Importance (sorted by absolute coefficient):")
print(feature_importance)
print()

# ============================================================================
# 6. MAKE PREDICTIONS
# ============================================================================
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Sample predictions on test set:")
for i in range(min(5, len(X_test))):
    print(f"  Actual: {y_test[i]:6.2f}, Predicted: {y_test_pred[i]:6.2f}")
print()

# ============================================================================
# 7. EVALUATE THE MODEL
# ============================================================================
# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Testing metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Model Evaluation Metrics:")
print("\nTraining Set:")
print(f"  MSE: {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTest Set:")
print(f"  MSE: {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")
print()

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature Importance
ax1 = axes[0, 0]
colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
ax1.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
ax1.set_xlabel('Coefficient Value')
ax1.set_title('Feature Coefficients (Importance)')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = axes[0, 1]
ax2.scatter(y_test, y_test_pred, alpha=0.6)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Actual vs Predicted (R² = {test_r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Distribution
ax3 = axes[1, 0]
residuals = y_test - y_test_pred
ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Residuals Distribution')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals vs Predicted
ax4 = axes[1, 1]
ax4.scatter(y_test_pred, residuals, alpha=0.6)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title('Residual Plot')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'multiple_linear_regression_analysis.png'")
plt.show()

# ============================================================================
# 9. MODEL INTERPRETATION
# ============================================================================
print("\nModel Interpretation:")
print(f"The model explains {test_r2*100:.2f}% of the variance in the test data")
print(f"Most important positive feature: {feature_importance.iloc[0]['Feature']} (coef: {feature_importance.iloc[0]['Coefficient']:.4f})")
print(f"Most important negative feature: {feature_importance.iloc[-1]['Feature']} (coef: {feature_importance.iloc[-1]['Coefficient']:.4f})")
print()
print("Notes:")
print("- Positive coefficients indicate features that increase the target value")
print("- Negative coefficients indicate features that decrease the target value")
print("- Larger absolute values indicate stronger relationships")
print("- The model is evaluated on scaled features for fair comparison")
