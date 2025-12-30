#!/usr/bin/env python3
"""
Simple Linear Regression Implementation and Demonstration

This script demonstrates:
- How Simple Linear Regression works
- Data preparation and visualization
- Model training and evaluation
- Prediction on new data
- Error analysis and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. GENERATE SYNTHETIC DATA
# ============================================================================
np.random.seed(42)
n_samples = 100

# Create a simple linear relationship: y = 2.5*x + noise
X = np.random.randn(n_samples, 1) * 10
y = 2.5 * X.ravel() + np.random.randn(n_samples) * 5

print("Dataset Information:")
print(f"Number of samples: {n_samples}")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print()

# ============================================================================
# 2. SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print()

# ============================================================================
# 3. CREATE AND TRAIN THE MODEL
# ============================================================================
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Parameters:")
print(f"Slope (coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Equation: y = {model.coef_[0]:.4f}*x + {model.intercept_:.4f}")
print()

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Sample predictions on test set:")
for i in range(min(5, len(X_test))):
    print(f"X={X_test[i][0]:.2f}, Actual={y_test[i]:.2f}, Predicted={y_test_pred[i]:.2f}")
print()

# ============================================================================
# 5. EVALUATE THE MODEL
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
print(f"  MSE (Mean Squared Error): {train_mse:.4f}")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.4f}")
print(f"  MAE (Mean Absolute Error): {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTest Set:")
print(f"  MSE (Mean Squared Error): {test_mse:.4f}")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.4f}")
print(f"  MAE (Mean Absolute Error): {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")
print()

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Data with Regression Line
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha=0.6, label='Training Data')
ax1.plot(X_train, y_train_pred, 'r-', linewidth=2, label='Regression Line')
ax1.set_xlabel('X (Independent Variable)')
ax1.set_ylabel('y (Dependent Variable)')
ax1.set_title('Training Data with Fitted Line')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Test Data with Predictions
ax2 = axes[0, 1]
ax2.scatter(X_test, y_test, alpha=0.6, label='Test Data')
ax2.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Predictions')
ax2.set_xlabel('X (Independent Variable)')
ax2.set_ylabel('y (Dependent Variable)')
ax2.set_title('Test Data with Predictions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = y_test - y_test_pred
ax3.scatter(y_test_pred, residuals, alpha=0.6)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted
ax4 = axes[1, 1]
ax4.scatter(y_test, y_test_pred, alpha=0.6)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Values')
ax4.set_ylabel('Predicted Values')
ax4.set_title(f'Actual vs Predicted (R² = {test_r2:.4f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'simple_linear_regression_analysis.png'")
plt.show()

# ============================================================================
# 7. MAKE PREDICTIONS ON NEW DATA
# ============================================================================
new_X = np.array([[5], [10], [15], [20]])
new_predictions = model.predict(new_X)

print("\nPredictions on New Data:")
for x_val, pred in zip(new_X.ravel(), new_predictions):
    print(f"  When X = {x_val:3.0f}: Predicted y = {pred:7.2f}")
print()

# ============================================================================
# 8. MATHEMATICAL INTERPRETATION
# ============================================================================
print("Mathematical Interpretation:")
print(f"For every 1 unit increase in X, y increases by {model.coef_[0]:.4f} units")
print(f"When X = 0, the predicted y value (intercept) is {model.intercept_:.4f}")
print(f"The R² score of {test_r2:.4f} means the model explains {test_r2*100:.2f}% of the variance in the data")
