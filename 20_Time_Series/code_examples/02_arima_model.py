from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ARIMA Model Implementation

def fit_arima_model(ts, order=(1, 1, 1)):
    """
    Fit ARIMA model to time series
    """
    model = ARIMA(ts, order=order)
    results = model.fit()
    return results

def make_predictions(model, steps=10):
    """
    Make forecast predictions
    """
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

def evaluate_arima(actual, predicted):
    """
    Evaluate ARIMA model performance
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'RMSE': rmse}

if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.Series(np.random.randn(100).cumsum(), index=dates)
    
    # Fit model
    model = fit_arima_model(data)
    
    # Make predictions
    forecast, conf_int = make_predictions(model, steps=10)
    print(f"Forecast: {forecast.values}")
    print(f"Model Summary:\n{model.summary()}")
