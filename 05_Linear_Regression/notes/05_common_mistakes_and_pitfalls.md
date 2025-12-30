# Common Mistakes and Pitfalls in Linear Regression

## 1. Not Checking Assumptions

### The Mistake
Applying linear regression without verifying that the underlying assumptions are met.

### Key Assumptions to Verify
1. **Linearity**: Relationship between X and y is linear
   - Check: Scatter plots, residual plots
   - Fix: Add polynomial features, transformations

2. **Independence**: Observations are independent
   - Check: Autocorrelation plots, time series analysis
   - Fix: Account for temporal patterns, use time-series methods

3. **Homoscedasticity**: Error variance is constant
   - Check: Residual vs fitted plot
   - Fix: Use weighted least squares, transformations

4. **Normality**: Residuals are normally distributed
   - Check: Q-Q plot, Shapiro-Wilk test
   - Fix: Transformations, robust regression methods

### Why It Matters
Violated assumptions lead to invalid statistical inference and unreliable predictions.

**YouTube Reference**: Search "Linear Regression Assumption Testing" for tutorials

## 2. Forgetting to Scale Features

### The Problem
```python
# WRONG - Using raw, unscaled features
model = LinearRegression()
model.fit(X_raw, y)  # X might have features in range [0, 1000000]
```

### Why This Fails
- Gradient descent converges very slowly
- Coefficients become hard to interpret
- Regularization (Ridge/Lasso) works ineffectively
- Numerical instability in matrix computations

### The Solution
```python
# CORRECT - Scale features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
```

### Important Rule
**ALWAYS scale features before using gradient-based methods or regularization**

## 3. Not Handling Missing Data Properly

### Common Mistakes
1. **Deleting entire rows with any missing value**
   - Loss of valuable information
   - Can introduce bias if data is not missing completely at random (MCAR)

2. **Using simple mean imputation everywhere**
   - Reduces variance artificially
   - Creates inconsistent patterns
   - Not suitable for tree-based methods that follow

3. **Ignoring the "missingness" pattern**
   - Sometimes missingness itself is informative
   - Could indicate data quality issues

### Proper Approach
```python
# For MCAR data, use appropriate methods:
from sklearn.impute import SimpleImputer, KNNImputer

# Option 1: KNN Imputation (better for correlated features)
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Option 2: Multiple Imputation (preserves uncertainty)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)
```

## 4. Data Leakage

### The Mistake
Including information in features that wouldn't be available at prediction time.

### Example 1: Scaling Leak
```python
# WRONG - Fit scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # LEAK!
X_train, X_test = train_test_split(X_scaled, ...)

# CORRECT - Fit scaler only on training data
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### Example 2: Future Information Leak
```python
# WRONG - Using future values as features
df['price_momentum'] = df['future_price'] - df['current_price']  # LEAK!

# CORRECT - Use only historical information
df['price_momentum'] = df['current_price'] - df['previous_price']
```

### Why It Matters
- Overly optimistic performance estimates
- Model fails in production
- False confidence in model capabilities

## 5. Overfitting to Training Data

### Signs of Overfitting
- Training R² = 0.98, Test R² = 0.65 (large gap)
- Model captures noise instead of signal
- Too many features relative to samples

### Prevention Strategies
1. **Use Cross-Validation**
   - K-fold CV instead of single train-test split
   - Provides robust performance estimates

2. **Regularization**
   ```python
   from sklearn.linear_model import Ridge, Lasso
   # Ridge: All features, shrunk coefficients
   ridge = Ridge(alpha=1.0)
   # Lasso: Feature selection, some coefficients = 0
   lasso = Lasso(alpha=0.1)
   ```

3. **Feature Selection**
   - Keep only important features
   - Use domain knowledge
   - Statistical tests (p-values, correlation)

## 6. Not Validating on Proper Test Set

### Mistakes
1. **Using training data for evaluation**
   - Always use held-out test set
   - Data the model has never seen

2. **No temporal validation for time series**
   ```python
   # WRONG
   X_train, X_test = train_test_split(X_timeseries, shuffle=True)  # BAD!
   
   # CORRECT
   split_point = int(len(X) * 0.8)
   X_train, X_test = X[:split_point], X[split_point:]
   ```

3. **Test set contamination**
   - Test set must remain truly held-out
   - Never modify based on test performance

## 7. Multicollinearity Issues

### The Problem
Highly correlated features cause:
- Unstable, unreliable coefficients
- Small changes in data lead to large coefficient changes
- Difficult to interpret feature importance

### Detection
```python
# Method 1: Correlation matrix
import seaborn as sns
sns.heatmap(X.corr(), annot=True)  # Look for values > 0.8

# Method 2: Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# VIF > 10 indicates multicollinearity
```

### Solutions
1. **Drop one of the correlated features**
2. **Use Ridge Regression** (handles multicollinearity)
3. **PCA** (Principal Component Analysis) for dimensionality reduction
4. **Domain knowledge** to guide feature selection

## 8. Ignoring Outliers

### The Risk
Outliers can:
- Disproportionately influence regression line
- Violate normality assumption
- Indicate data quality issues

### Detection
```python
# Method 1: Statistical (Standardized residuals > 3)
residuals = y_actual - y_predicted
std_residuals = residuals / residuals.std()
outliers = np.abs(std_residuals) > 3

# Method 2: Cook's Distance
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model_result)
cooks_d = influence.cooks_distance[0]
# Values > 4/n indicate influential points
```

### Action Plan
1. **Investigate first** - Is it a real observation or data error?
2. **Document** - Record why outlier was treated
3. **Don't delete automatically** - Outliers can be valuable information
4. **Try both** - Fit with and without outliers

## 9. Using Wrong Evaluation Metrics

### Common Mistakes
1. **Using R² for small sample sizes**
   - R² is biased upward
   - Use Adjusted R² instead

2. **Only looking at one metric**
   ```python
   # Check multiple metrics
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   
   mse = mean_squared_error(y_true, y_pred)
   rmse = np.sqrt(mse)  # Same units as y
   mae = mean_absolute_error(y_true, y_pred)
   r2 = r2_score(y_true, y_pred)
   ```

3. **Not considering domain-specific costs**
   - Underestimating future demand = lost sales
   - Overestimating future demand = excess inventory
   - These have different business impacts

## 10. Not Documenting Preprocessing Steps

### The Problem
- Hard to reproduce model in production
- Difficult for others to understand your work
- Easy to make mistakes when applying to new data

### Solution: Document Everything
```python
# Keep a preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Now preprocessing and model are coupled
X_train_processed = pipeline.named_steps['scaler'].fit_transform(X_train)
y_pred = pipeline.predict(X_test)  # Scaler already applied internally
```

## Quick Checklist

Before deploying a linear regression model:

- [ ] Verified all assumptions (linearity, independence, homoscedasticity, normality)
- [ ] Scaled all features appropriately
- [ ] Handled missing data properly (with documentation)
- [ ] Checked for data leakage
- [ ] Used proper train-test split (stratified if needed)
- [ ] Validated with cross-validation
- [ ] Checked for multicollinearity
- [ ] Investigated outliers and documented treatment
- [ ] Used appropriate evaluation metrics
- [ ] Documented preprocessing pipeline
- [ ] Validated with domain experts
- [ ] Tested on truly held-out data

## Resources

**YouTube**: "Linear Regression Mistakes" and "Regression Diagnostics"
**Books**: "Applied Predictive Modeling" by Kuhn & Johnson
**Papers**: Original regression assumption papers for theoretical depth
