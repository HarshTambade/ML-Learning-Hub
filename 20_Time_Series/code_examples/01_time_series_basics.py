import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Time series basics demonstration

def create_sample_time_series(n_points=100):
    """
    Create a sample time series with trend and seasonality
    """
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    
    # Create components
    trend = np.linspace(0, 10, n_points)
    seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.normal(0, 1, n_points)
    
    values = trend + seasonality + noise
    
    return pd.Series(values, index=dates, name='Time Series Data')

def analyze_time_series(ts):
    """
    Analyze basic properties of time series
    """
    print("Time Series Analysis")
    print(f"Length: {len(ts)}")
    print(f"Start Date: {ts.index[0]}")
    print(f"End Date: {ts.index[-1]}")
    print(f"Mean: {ts.mean():.4f}")
    print(f"Std Dev: {ts.std():.4f}")
    print(f"Min: {ts.min():.4f}")
    print(f"Max: {ts.max():.4f}")

def visualize_time_series(ts):
    """
    Visualize time series data
    """
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values, linewidth=2)
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def resample_time_series(ts, frequency='W'):
    """
    Resample time series to different frequency
    """
    return ts.resample(frequency).mean()

if __name__ == "__main__":
    # Create time series
    ts = create_sample_time_series()
    
    # Analyze
    analyze_time_series(ts)
    
    # Visualize
    visualize_time_series(ts)
    
    # Resample
    weekly = resample_time_series(ts, 'W')
    print("\nWeekly Resampled Series:")
    print(weekly.head())
