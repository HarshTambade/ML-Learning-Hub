# Model Diagnostics and Validation

## Residual Analysis

### 1. Residuals Plot
- Plot residuals vs fitted values
- Look for patterns: ideally random scatter
- Patterns indicate violations of assumptions

### 2. Q-Q Plot
- Check if residuals are normally distributed
- Points should follow the diagonal line
- Deviations indicate non-normality

### 3. Scale-Location Plot
- Shows sqrt(standardized residuals) vs fitted values
- Constant slope indicates homoscedasticity
- Increasing/decreasing slope indicates heteroscedasticity

## Assumption Checks

### Linearity
- Visual: scatter plot with regression line
- Test: residuals should have mean = 0

### Independence
- Durbin-Watson test (value near 2 is good)
- Autocorrelation function plot

### Homoscedasticity (Constant Variance)
- Breusch-Pagan test
- Visual inspection of residual plot

### Normality
- Shapiro-Wilk test (p > 0.05 = normal)
- Anderson-Darling test
- Kolmogorov-Smirnov test

## Outlier Detection

### Cook's Distance
- Identifies influential points
- Values > 4/n suggest influential observations
- Remove if data entry error, else consider keeping

### Standardized Residuals
- |residual| > 3 suggests outlier
- Investigate before removal

## Multicollinearity Check

### Correlation Matrix
- Check for correlations > 0.8
- High correlation causes unstable coefficients

### Variance Inflation Factor (VIF)
- VIF > 10 indicates high multicollinearity
- VIF > 5 warrants investigation

## Model Selection Criteria

### AIC (Akaike Information Criterion)
- Lower AIC = better balance of fit and complexity
- Penalizes number of parameters

### BIC (Bayesian Information Criterion)
- Similar to AIC but stronger penalty
- Preferred when sample size is large

### Adjusted R²
- Accounts for number of predictors
- Decreases if adding non-useful variables
