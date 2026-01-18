from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# LSTM Time Series Forecasting

def prepare_data(data, lookback=10):
    """
    Prepare data for LSTM model
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def build_lstm_model(lookback=10):
    """
    Build LSTM neural network model
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, X_train, y_train, epochs=50):
    """
    Train LSTM model
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return history

def forecast_lstm(model, last_sequence, steps=10):
    """
    Make LSTM forecasts
    """
    predictions = []
    current = last_sequence.copy()
    for _ in range(steps):
        next_pred = model.predict(current.reshape(1, -1, 1))[0, 0]
        predictions.append(next_pred)
        current = np.append(current[1:], next_pred)
    return np.array(predictions)

if __name__ == "__main__":
    # Create sample data
    data = np.random.randn(100).cumsum()
    
    # Prepare data
    X, y = prepare_data(data, lookback=10)
    
    # Build and train model
    model = build_lstm_model(lookback=10)
    train_lstm(model, X, y, epochs=20)
    
    # Forecast
    forecast = forecast_lstm(model, data[-10:], steps=10)
    print(f"LSTM Forecast: {forecast}")
