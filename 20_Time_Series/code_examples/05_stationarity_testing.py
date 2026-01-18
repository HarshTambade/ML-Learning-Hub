from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Stationarity Testing for Time Series

def adf_test(ts, name=''):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(ts, autolag='AIC')
    print(f'ADF Test Results for {name}:')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Stationary: {"Yes" if result[1] < 0.05 else "No"}')
    return result[1] < 0.05

def kpss_test(ts, name=''):
    """
    Perform KPSS test for stationarity
    """
    result = kpss(ts, regression='c', nlags='auto')
    print(f'KPSS Test Results for {name}:')
    print(f'KPSS Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Stationary: {"Yes" if result[1] > 0.05 else "No"}')
    return result[1] > 0.05

def difference_series(ts, order=1):
    """
    Apply differencing to make series stationary
    """
    diff = ts.diff().dropna()
    for i in range(order-1):
        diff = diff.diff().dropna()
    return diff

if __name__ == "__main__":
    # Create non-stationary series
    ts = pd.Series(np.random.randn(100).cumsum())
    
    # Test original series
    adf_test(ts, 'Original')
    kpss_test(ts, 'Original')
    
    # Difference and test
    diff_ts = difference_series(ts, order=1)
    adf_test(diff_ts, 'Differenced')
    kpss_test(diff_ts, 'Differenced')
