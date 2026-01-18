# Exponential Smoothing - Detailed Notes

## Exponential Smoothing Overview
Weights past observations exponentially, with more weight on recent data.

## Simple Exponential Smoothing (SES)
`y^(t+1) = α*y(t) + (1-α)*y^(t)`

**α parameter**:
- Range: 0-1
- High α (0.8-0.99): Reactive to changes
- Low α (0.01-0.1): Smooth, less reactive

## Double Exponential Smoothing (Holt's)
Incorporates trend:
- Level: `L(t) = α*y(t) + (1-α)(L(t-1) + T(t-1))`
- Trend: `T(t) = β*(L(t) - L(t-1)) + (1-β)*T(t-1)`
- Forecast: `y^(t+h) = L(t) + h*T(t)`

## Triple Exponential Smoothing (Holt-Winters)
Adds seasonality:
- Level
- Trend  
- Seasonality components

**Variants**:
- Additive: For constant seasonal magnitude
- Multiplicative: For increasing seasonal magnitude

## Advantages
- Simple to implement
- Fast computation
- Good for short-term forecasts
- Adaptive to recent changes

## Limitations
- Requires manual parameter tuning
- Assumes no structural breaks
- Poor for long-term forecasts
