#!/usr/bin/env python3
"""
Polynomial Regression Implementation and Demonstration

This script demonstrates:
- Non-linear relationships using polynomial features
- Overfitting and underfitting concepts
- Degree selection and model complexity
- Comparison of linear vs polynomial models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# 1. GENERATE SYNTHETIC NON-LINEAR DATA
# ============================================================================
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * (X**2).ravel() - 3 * X.ravel() + 2 + np.random.normal(0, 10, 100)

print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
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
# 3. COMPARE DIFFERENT POLYNOMIAL DEGREES
# ============================================================================
results = []
degrees = [1, 2, 3, 4, 5]

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    results.append({
        'Degree': degree,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Model': model,
        'Poly': poly_features,
        'y_pred': y_test_pred
    })
    
    print(f"Degree {degree}:")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print()

# ============================================================================
# 4. SELECT BEST MODEL
# ============================================================================
best_result = max(results, key=lambda x: x['Test R²'])
best_degree = best_result['Degree']
best_model = best_result['Model']
best_poly = best_result['Poly']

print(f"Best model: Polynomial degree {best_degree}")
print(f"Best Test R²: {best_result['Test R²']:.4f}")
print()

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Comparison of polynomial degrees
ax1 = axes[0, 0]
for result in results:
    ax1.plot(result['Degree'], result['Test R²'], 'o-', markersize=8, label=f"Degree {result['Degree']}")
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Test R² Score')
ax1.set_title('Model Performance vs Polynomial Degree')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(degrees)

# Plot 2: Train vs Test R²
ax2 = axes[0, 1]
degs = [r['Degree'] for r in results]
train_r2s = [r['Train R²'] for r in results]
test_r2s = [r['Test R²'] for r in results]
ax2.plot(degs, train_r2s, 'o-', linewidth=2, markersize=8, label='Train R²')
ax2.plot(degs, test_r2s, 's-', linewidth=2, markersize=8, label='Test R²')
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('R² Score')
ax2.set_title('Training vs Testing Performance')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(degrees)

# Plot 3: Best Model Fit
ax3 = axes[1, 0]
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_plot_poly = best_poly.transform(X_plot)
y_plot = best_model.predict(X_plot_poly)
ax3.scatter(X_test, y_test, alpha=0.6, label='Test Data')
ax3.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {best_degree} Fit')
ax3.set_xlabel('X')
ax3.set_ylabel('y')
ax3.set_title(f'Best Polynomial Fit (Degree {best_degree})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals
ax4 = axes[1, 1]
residuals = y_test - best_result['y_pred']
ax4.scatter(best_result['y_pred'], residuals, alpha=0.6)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title('Residual Plot (Best Model)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_regression_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'polynomial_regression_analysis.png'")
plt.show()

# ============================================================================
# 6. OVERFITTING AND UNDERFITTING ANALYSIS
# ============================================================================
print("\n=== Overfitting and Underfitting Analysis ===")
underfitting = min(results, key=lambda x: x['Test R²'])
overfitting = max(results, key=lambda x: abs(x['Train R²'] - x['Test R²']))

print(f"\nUnerfitting Example (Degree {underfitting['Degree']}):")
print(f"  Train R²: {underfitting['Train R²']:.4f}")
print(f"  Test R²: {underfitting['Test R²']:.4f}")
print(f"  Gap: {underfitting['Train R²'] - underfitting['Test R²']:.4f}")

print(f"\nOverfitting Example (Degree {overfitting['Degree']}):")
print(f"  Train R²: {overfitting['Train R²']:.4f}")
print(f"  Test R²: {overfitting['Test R²']:.4f}")
print(f"  Gap: {overfitting['Train R²'] - overfitting['Test R²']:.4f}")

print(f"\nBest Balance (Degree {best_degree}):")
print(f"  Train R²: {best_result['Train R²']:.4f}")
print(f"  Test R²: {best_result['Test R²']:.4f}")
print(f"  Gap: {best_result['Train R²'] - best_result['Test R²']:.4f}")

print("\nKey Insights:")
print("- Degree 1 (Linear): Usually underfits non-linear data")
print("- Higher degrees: Can overfit if gap between train/test R² is large")
print("- Best model: Balances low bias with low variance")
print("- Monitor both training and testing metrics to detect overfitting")
