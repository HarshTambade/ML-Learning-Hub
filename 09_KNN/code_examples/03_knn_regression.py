#!/usr/bin/env python3
"""
KNN Regression Example

Demonstrates KNN for regression problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_regression_data():
    X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_knn_regressors(X_train, X_test, y_train, y_test):
    print("="*50)
    print("KNN REGRESSION EXAMPLE")
    print("="*50 + "\n")
    
    k_values = [1, 3, 5, 7, 10]
    results = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        results[k] = {'mse': mse, 'mae': mae, 'r2': r2, 'model': knn}
        
        print(f"K = {k}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}\n")
    
    return results, X_train, X_test, y_train, y_test

def visualize_regression(results, X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: K vs MSE
    k_vals = list(results.keys())
    mse_vals = [results[k]['mse'] for k in k_vals]
    axes[0, 0].plot(k_vals, mse_vals, 'o-', color='red')
    axes[0, 0].set_title('MSE vs K Value')
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: K vs R2
    r2_vals = [results[k]['r2'] for k in k_vals]
    axes[0, 1].plot(k_vals, r2_vals, 'o-', color='blue')
    axes[0, 1].set_title('R² Score vs K Value')
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs Actual (K=3)
    best_model = results[3]['model']
    y_pred = best_model.predict(X_test)
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_title('Predicted vs Actual (K=3)')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Data and fit curve
    X_range = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
    y_range = best_model.predict(X_range)
    axes[1, 1].scatter(X_train, y_train, label='Training Data')
    axes[1, 1].plot(X_range, y_range, 'r-', label='KNN (K=3)', linewidth=2)
    axes[1, 1].set_title('KNN Regression Fit')
    axes[1, 1].set_xlabel('Feature')
    axes[1, 1].set_ylabel('Target')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_regression.png', dpi=100, bbox_inches='tight')
    print("Visualization saved as 'knn_regression.png'")
    plt.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_regression_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results, X_train, X_test, y_train, y_test = train_knn_regressors(X_train, X_test, y_train, y_test)
    visualize_regression(results, X_train, X_test, y_train, y_test)
    
    print("="*50)
    print("REGRESSION COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")
