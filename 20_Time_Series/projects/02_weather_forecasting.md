# Project 2: Weather Forecasting

## Objective
Forecast temperature, humidity, and precipitation for next 7 days.

## Data
- Historical weather data (3+ years)
- Features: Temperature, Humidity, Pressure, Wind Speed, Precipitation
- Granularity: Hourly or daily data
- Source: Weather APIs or datasets

## Tasks
1. **Data Preparation**
   - Load and clean weather data
   - Handle missing values
   - Create rolling features (averages, max/min)

2. **Feature Engineering**
   - Seasonal indicators
   - Holiday markers
   - Lagged features

3. **Model Development**
   - Separate models for each variable
   - ARIMA/SARIMA for seasonal patterns
   - LSTM for temperature

4. **Multi-step Forecasting**
   - Forecast 1-7 days ahead
   - Compare strategies (recursive vs direct)

## Deliverables
- 7-day forecast
- Accuracy metrics for each variable
- Visualization of predictions
- Confidence intervals

## Challenges
- Weather has strong seasonal patterns
- Multiple correlated variables
- External factors (climate changes)
