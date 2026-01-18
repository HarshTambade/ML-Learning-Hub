# Project 1: Stock Price Forecasting

## Objective
Forecast daily closing prices using multiple time series models.

## Data
- Historical daily stock data (5+ years recommended)
- Features: Open, High, Low, Close, Volume
- Data source: yfinance, Alpha Vantage, or CSV

## Tasks
1. **Data Collection & Preprocessing**
   - Load historical data
   - Handle missing values
   - Normalize/Scale features

2. **Exploratory Analysis**
   - Plot price trends
   - Analyze volatility
   - Check for seasonality/patterns

3. **Model Development**
   - ARIMA model
   - LSTM neural network
   - Exponential smoothing
   - Compare performance

4. **Evaluation**
   - 80-20 or 70-30 train-test split
   - Calculate MAE, RMSE, MAPE
   - Visualize predictions

## Deliverables
- Trained models
- Performance comparison report
- Visualization of predictions
- Discussion of limitations

## Key Considerations
- Stock prices are non-stationary
- Apply differencing if needed
- External factors affect prices
- Use ensemble methods for better accuracy
