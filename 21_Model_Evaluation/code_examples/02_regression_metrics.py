"""Regression Metrics - MAE, MSE, RMSE, R-squared

Comprehensive guide to evaluating regression model performance
with practical implementations and visualizations.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results[name] = {'y_pred': y_pred, 'MAE': mae, 'MSE': mse, 
                     'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    
    print(f"\n{name}:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Predictions vs Actual
for idx, (name, result) in enumerate(results.items()):
    axes[0, 0].scatter(y_test, result['y_pred'], alpha=0.5, label=name)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Predictions vs Actual')
axes[0, 0].legend()

# Residuals
for idx, (name, result) in enumerate(results.items()):
    residuals = y_test - result['y_pred']
    axes[0, 1].scatter(result['y_pred'], residuals, alpha=0.5, label=name)
axes[0, 1].axhline(y=0, color='k', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].legend()

# Metrics Comparison
metrics = list(results[list(results.keys())[0]].keys())[1:]
for metric in metrics:
    values = [results[name][metric] for name in results.keys()]
    axes[1, 0].plot(list(results.keys()), values, marker='o', label=metric)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Metrics Comparison')
axes[1, 0].legend()

# R-squared Comparison
models_list = list(results.keys())
r2_values = [results[name]['R2'] for name in models_list]
axes[1, 1].bar(models_list, r2_values)
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('R² Score Comparison')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.show()

print("\nRegression metrics evaluation complete!")
