# Time Series Fundamentals - Detailed Notes

## What is a Time Series?
A time series is a sequence of data points indexed in time order. Key characteristics:
- **Temporal ordering**: Data points follow chronological order
- **Dependencies**: Observations are not independent; past values influence future values
- **Patterns**: Contains trend, seasonality, cyclical patterns
- **Stationarity**: May require transformation to become stationary

## Components of Time Series

### 1. Trend Component
- **Definition**: Long-term movement in the data
- **Types**: Increasing, decreasing, or constant
- **Detection**: Visual inspection or statistical tests
- **Impact**: Affects forecasting accuracy
- **Method to handle**: Differencing or detrending

### 2. Seasonality Component
- **Definition**: Repeating patterns at regular intervals
- **Period**: Daily, weekly, monthly, yearly
- **Characteristics**: Regular and predictable
- **Impact**: Important for accurate forecasts
- **Measurement**: Seasonal index calculation

### 3. Cyclical Component
- **Definition**: Long-term oscillations not at fixed interval
- **Duration**: Longer than seasonal patterns
- **Causes**: Economic, business cycles
- **Difference from seasonality**: No fixed period

### 4. Residual/Noise Component
- **Definition**: Irregular fluctuations
- **Characteristics**: Random, unpredictable
- **Impact**: Limits forecast accuracy
- **Analysis**: Should be white noise

## Stationarity
**Definition**: Statistical properties don't change over time

**Requirements for Stationarity**:
- Constant mean
- Constant variance
- Autocovariance independent of time

**Why Important**:
- Most forecasting models assume stationarity
- Non-stationary data leads to spurious correlations
- Transforms required for modeling

**Tests for Stationarity**:
- Augmented Dickey-Fuller (ADF) Test
- KPSS Test
- Phillips-Perron Test

## Autocorrelation and Partial Autocorrelation

**ACF (Autocorrelation Function)**:
- Measures correlation between observations at different lags
- Useful for identifying MA (Moving Average) order
- Plot interpretation guides model selection

**PACF (Partial Autocorrelation Function)**:
- Correlation after removing intermediate lags
- Useful for identifying AR (Autoregressive) order
- Helps in model parameter selection

## Common Time Series Patterns

1. **Uptrend**: Consistently increasing values
2. **Downtrend**: Consistently decreasing values  
3. **Seasonal Pattern**: Repeating cycles
4. **Random Walk**: Unpredictable movements
5. **Mean Reverting**: Fluctuates around a mean

## Time Series Analysis Steps

1. **Data Collection**: Gather appropriate historical data
2. **Data Visualization**: Plot and explore patterns
3. **Stationarity Testing**: Check and transform if needed
4. **Model Selection**: Choose appropriate model
5. **Parameter Estimation**: Fit model to data
6. **Diagnostics**: Check residuals
7. **Forecasting**: Make predictions
8. **Evaluation**: Assess forecast accuracy

## Key Metrics for Evaluation
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- AIC/BIC (Information Criteria)
