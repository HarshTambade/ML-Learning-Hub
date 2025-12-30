# Linear Regression Exercises

## Exercise 1: Simple Linear Regression from Scratch

**Problem**: Implement simple linear regression without using sklearn

**Objectives**:
- Calculate correlation coefficient
- Compute slope and intercept using formulas
- Make predictions
- Calculate R² score

**Dataset**: Boston Housing (or your own synthetic data)

**Solution Approach**:
```python
import numpy as np

def simple_linear_regression(X, y):
    n = len(X)
    mean_X = np.mean(X)
    mean_y = np.mean(y)
    
    # Calculate slope
    numerator = np.sum((X - mean_X) * (y - mean_y))
    denominator = np.sum((X - mean_X) ** 2)
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = mean_y - slope * mean_X
    
    return slope, intercept
```

**Key Concepts to Verify**:
- Slope represents rate of change
- Intercept is y-value when X=0
- Use same formulas as sklearn LinearRegression

---

## Exercise 2: Feature Scaling Impact

**Problem**: Demonstrate the importance of feature scaling

**Task**:
1. Train linear regression on raw (unscaled) features
2. Train on standardized features
3. Compare results

**Expected Outcome**:
- Models should give same predictions
- Coefficients differ due to scaling
- Different numerical stability

**Code Template**:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Raw data
model_raw = LinearRegression()
model_raw.fit(X_raw, y)

# Scaled data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

# Compare predictions
print("Predictions match:", np.allclose(model_raw.predict(X_raw), model_scaled.predict(X_scaled)))
```

---

## Exercise 3: Assumption Validation

**Problem**: Check if linear regression assumptions hold for your data

**Assumptions to Test**:
1. **Linearity**: Use scatter plot with regression line
2. **Independence**: Check Durbin-Watson statistic (should be ~2)
3. **Homoscedasticity**: Plot residuals vs fitted values
4. **Normality**: Q-Q plot and Shapiro-Wilk test

**Code**:
```python
import matplotlib.pyplot as plt
from scipy import stats

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

# 1. Linearity
plt.scatter(X, y)
plt.plot(X, y_pred, 'r-')
plt.show()

# 2. Independence (Durbin-Watson)
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"Durbin-Watson: {dw} (should be ~2)")

# 3. Homoscedasticity
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# 4. Normality
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value} (>0.05 suggests normality)")
```

---

## Exercise 4: Polynomial Regression

**Problem**: Build polynomial regression models of different degrees

**Task**:
1. Generate non-linear synthetic data
2. Fit polynomials of degree 1-5
3. Evaluate each model
4. Identify overfitting

**Evaluation Metrics**:
- Training R²
- Test R²
- MSE
- Identify where gap widens (overfitting starts)

**Code Template**:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

for degree in range(1, 6):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    train_r2 = model.score(X_poly, y_train)
    test_r2 = model.score(poly.transform(X_test), y_test)
    
    print(f"Degree {degree}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
```

---

## Exercise 5: Regularization Comparison

**Problem**: Compare Ridge, Lasso, and ElasticNet

**Objectives**:
1. Train models with different alpha values
2. Compare feature selection (Lasso)
3. Evaluate on test set
4. Visualize coefficient paths

**Key Differences**:
- Ridge: Shrinks all coefficients
- Lasso: Some coefficients become exactly 0
- ElasticNet: Hybrid approach

**Code**:
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    elasticnet = ElasticNet(alpha=alpha)
    
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    elasticnet.fit(X_train, y_train)
    
    print(f"\nAlpha = {alpha}")
    print(f"Ridge R²: {ridge.score(X_test, y_test):.4f}")
    print(f"Lasso R²: {lasso.score(X_test, y_test):.4f} (Non-zero coefs: {np.sum(lasso.coef_ != 0)})")
    print(f"ElasticNet R²: {elasticnet.score(X_test, y_test):.4f}")
```

---

## Exercise 6: Data Leakage Detection

**Problem**: Identify and fix data leakage issues

**Scenarios**:  
1. **Scaling Leakage**: Fit scaler on entire dataset before split
2. **Feature Leakage**: Include target information in features
3. **Temporal Leakage**: Use future data for predictions

**Task**: Refactor leaky code to proper implementation

**Wrong Code**:
```python
# WRONG - Leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on entire dataset
X_train, X_test = train_test_split(X_scaled, test_size=0.2)
```

**Correct Code**:
```python
X_train, X_test = train_test_split(X, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler.transform(X_test)  # Transform test
```

---

## Exercise 7: Multicollinearity Analysis

**Problem**: Detect and handle multicollinearity

**Methods**:
1. Correlation matrix
2. Variance Inflation Factor (VIF)
3. Eigenvalues of correlation matrix

**Code**:
```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Method 1: Correlation matrix
corr_matrix = X.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Method 2: VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data[vif_data['VIF'] > 5])  # VIF > 5 indicates multicollinearity
```

---

## Exercise 8: Cross-Validation Implementation

**Problem**: Implement k-fold cross-validation

**Task**:
1. Split data into 5 folds
2. Train model on each fold
3. Calculate metrics for each fold
4. Report mean and std of metrics

**Code**:
```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print(f"Cross-validation R² scores: {scores}")
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

---

## Exercise 9: Outlier Detection and Treatment

**Problem**: Identify and handle outliers

**Methods**:
1. Standardized residuals > 3
2. Cook's distance
3. Interquartile range (IQR)

**Decision Making**:
- Is it a data error? → Delete
- Is it real but rare? → Keep or use robust regression
- Is it valuable? → Keep and document

---

## Exercise 10: Real-world Prediction Task

**Problem**: Build and deploy a complete linear regression model

**Steps**:
1. Load and explore data
2. Handle missing values
3. Feature engineering
4. Train-test split
5. Scale features
6. Train model with cross-validation
7. Validate assumptions
8. Make predictions on new data
9. Document everything

**Dataset Suggestions**:
- Boston Housing
- California Housing  
- Real estate prices
- Stock prices
- Weather data

---

## Tips for Success

1. **Always visualize** your data first
2. **Check assumptions** before interpreting results
3. **Use cross-validation** for robust estimates
4. **Document your choices** - why did you scale? Why this alpha?
5. **Compare baseline** - is your model better than mean prediction?
6. **Validate on held-out data** - never use training data for evaluation
7. **Check predictions** - do they make sense in your domain?

---

## Additional Resources

- **YouTube**: Search "Linear Regression Python Tutorial"
- **Datasets**: Kaggle.com, UCI Machine Learning Repository
- **Documentation**: scikit-learn, statsmodels
- **Books**: "Hands-On Machine Learning" by Aurélien Géron
