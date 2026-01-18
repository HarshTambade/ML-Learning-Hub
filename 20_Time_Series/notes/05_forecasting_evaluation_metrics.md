# Forecasting Evaluation Metrics

## Error Metrics

### MAE (Mean Absolute Error)
`MAE = (1/n) ∑|y_i - ŷ_i|`
- Average absolute deviation
- Same units as y
- Not sensitive to outliers

### RMSE (Root Mean Squared Error)
`RMSE = sqrt((1/n) ∑(y_i - ŷ_i)^2)`
- Penalizes large errors
- Same units as y
- Sensitive to outliers

### MAPE (Mean Absolute Percentage Error)
`MAPE = (100/n) ∑|y_i - ŷ_i|/y_i %`
- Percentage error
- Scale-independent
- Issues with zero values

## Information Criteria

### AIC (Akaike Information Criterion)
`AIC = 2k - 2ln(L)`
- Balances fit and complexity
- Lower is better
- Penalizes additional parameters

### BIC (Bayesian Information Criterion)
`BIC = k*ln(n) - 2ln(L)`
- Stronger penalty for complexity
- Useful for model selection
- Tends to favor simpler models

## Theil's U Statistic
Compares forecasts to naive method
- U < 1: Better than naive
- U = 1: Same as naive
- U > 1: Worse than naive

## Model Selection Guidelines
1. Multiple metrics for comprehensive view
2. Domain-specific considerations
3. Test set performance priority
4. Cross-validation recommended
5. Residual analysis important
