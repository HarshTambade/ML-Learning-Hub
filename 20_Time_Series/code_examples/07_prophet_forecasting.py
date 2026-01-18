"""Prophet Forecasting - Facebook's Prophet Library

Prophet is a time series forecasting tool developed by Facebook that handles
seasonality, trend changes, and holiday effects automatically.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load or create sample data
def load_time_series_data():
    """
    Load example time series data
    """
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    values = np.cumsum(np.random.randn(365)) + 100
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    return df

# Initialize and fit Prophet model
def fit_prophet_model(df):
    """
    Fit Prophet model with yearly seasonality
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(df)
    return model

# Make forecasts
def make_prophet_forecast(model, periods=30):
    """
    Generate future forecasts
    """
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    return forecast

# Main execution
if __name__ == "__main__":
    # Load data
    df = load_time_series_data()
    
    # Fit model
    model = fit_prophet_model(df)
    
    # Make forecast
    forecast = make_prophet_forecast(model, periods=30)
    
    # Display forecast results
    print("Forecast Results:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
    
    # Plot results
    model.plot(forecast)
    plt.title('Prophet Time Series Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    
    # Components plot
    model.plot_components(forecast)
    plt.tight_layout()
    plt.show()
    
    print("Prophet forecasting completed successfully")
