from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Exponential Smoothing Implementation

def simple_exponential_smoothing(ts, alpha=0.2):
    """
    Apply simple exponential smoothing
    """
    model = SimpleExpSmoothing(ts)
    results = model.fit(smoothing_level=alpha)
    return results

def holt_exponential_smoothing(ts, trend='add'):
    """
    Apply Holt's exponential smoothing (with trend)
    """
    model = ExponentialSmoothing(ts, trend=trend)
    results = model.fit()
    return results

def holt_winters_smoothing(ts, seasonal='add', seasonal_periods=12):
    """
    Apply Holt-Winters smoothing (with trend and seasonality)
    """
    model = ExponentialSmoothing(ts, seasonal=seasonal, seasonal_periods=seasonal_periods)
    results = model.fit()
    return results

def forecast_smoothed(model, steps=10):
    """
    Forecast using smoothing model
    """
    return model.forecast(steps=steps)

if __name__ == "__main__":
    # Create sample data with trend and seasonality
    t = np.arange(100)
    data = 10 + 0.1*t + 5*np.sin(2*np.pi*t/12) + np.random.randn(100)
    ts = pd.Series(data)
    
    # Apply smoothing models
    ses = simple_exponential_smoothing(ts)
    hw = holt_winters_smoothing(ts)
    
    # Forecast
    forecast = forecast_smoothed(hw, steps=10)
    print(f"Forecast: {forecast.values}")
    print(f"SES Parameters: {ses.params}")
    print(f"H-W Summary:\n{hw.summary()}")
