"""Multivariate Time Series Forecasting

Multivariate time series analysis handles multiple related time series simultaneously.
Useful for forecasting when multiple variables influence the outcome.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate multivariate time series data
def create_multivariate_data():
    """
    Create sample multivariate time series data with multiple variables
    """
    np.random.seed(42)
    n_samples = 200
    
    # Create three related time series
    t = np.arange(n_samples)
    var1 = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    var2 = np.cos(0.05 * t) + var1 * 0.5 + np.random.normal(0, 0.1, n_samples)
    var3 = var1 + var2 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame({
        'var1': var1,
        'var2': var2,
        'var3': var3
    })
    return df

# Create lagged features for multivariate series
def create_lagged_features(df, n_lags=5):
    """
    Create lagged features for all variables
    """
    data = df.copy()
    
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            data[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Remove rows with NaN values
    data = data.dropna()
    return data

# Prepare train-test split
def prepare_data(df, n_lags=5, test_size=0.2):
    """
    Prepare data for multivariate forecasting
    """
    lagged_df = create_lagged_features(df, n_lags)
    
    # Features and targets
    target_cols = df.columns.tolist()
    feature_cols = [col for col in lagged_df.columns if col not in target_cols]
    
    X = lagged_df[feature_cols]
    y = lagged_df[target_cols]
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

# Train multivariate model
def train_multivariate_model(X_train, y_train):
    """
    Train multioutput regression model
    """
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

# Main execution
if __name__ == "__main__":
    # Create data
    df = create_multivariate_data()
    print(f"Data shape: {df.shape}")
    print(f"Variables: {df.columns.tolist()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, n_lags=5)
    
    # Train model
    model = train_multivariate_model(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    
    print("\nMultivariate Forecasting Results:")
    for i, col in enumerate(df.columns):
        print(f"{col} - MSE: {mse[i]:.4f}, MAE: {mae[i]:.4f}")
    
    # Plot predictions
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i, col in enumerate(df.columns):
        axes[i].plot(y_test.index, y_test.iloc[:, i], label='Actual', marker='o')
        axes[i].plot(y_test.index, y_pred[:, i], label='Predicted', marker='s')
        axes[i].set_title(f'{col} - Multivariate Forecast')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nMultivariate forecasting completed successfully")
