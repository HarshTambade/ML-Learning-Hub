# ARIMA Models Theory

## ARIMA Overview
AutoRegressive Integrated Moving Average - combines three techniques:
- AR (Autoregressive): Uses past values
- I (Integrated): Differencing for stationarity
- MA (Moving Average): Uses past errors

## Parameters: ARIMA(p,d,q)

**p (Autoregressive order)**:
- Number of lag observations
- Identified using PACF
- Default range: 0-5

**d (Integration/Differencing order)**:
- Number of times to difference for stationarity
- Usually 0, 1, or 2
- Test stationarity with ADF test

**q (Moving Average order)**:
- Size of moving average window
- Identified using ACF
- Default range: 0-5

## AR Component: AR(p)
Uses past values as predictors:
`y(t) = c + φ1*y(t-1) + φ2*y(t-2) + ... + εt`

## MA Component: MA(q)
Uses past forecast errors:
`y(t) = μ + εt + θ1*ε(t-1) + θ2*ε(t-2) + ...`

## I Component: Differencing
Makes series stationary:
- First difference: y'(t) = y(t) - y(t-1)
- Second difference: y''(t) = y'(t) - y'(t-1)

## Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)m
- Extensions for seasonal data
- P, D, Q for seasonal components
- m: seasonal period

## Model Selection
1. Check stationarity
2. Plot ACF/PACF
3. Test different (p,d,q)
4. Compare AIC/BIC values
5. Validate residuals

## Assumptions
- Stationarity after differencing
- No seasonality (use SARIMA if present)
- Residuals are white noise
- Constant variance
