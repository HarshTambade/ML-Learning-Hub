# Missing Value Imputation

## Introduction

Missing values are a common challenge in real-world datasets. Proper handling of missing data is crucial for model performance and data integrity.

## Types of Missing Data

### 1. Missing Completely at Random (MCAR)
- No pattern to the missing data
- Random mechanisms cause missing values

### 2. Missing at Random (MAR)
- Missingness depends on observed variables
- Can be addressed by imputation considering other variables

### 3. Missing Not at Random (MNAR)
- Missingness depends on unobserved variables
- Most problematic type

## Imputation Methods

### 1. Deletion Methods

**Pros**: Simple, removes bias
**Cons**: Loss of data

```python
# Remove rows with any missing values
df_dropped = df.dropna()

# Remove columns with missing values
df_dropped = df.dropna(axis=1)

# Remove rows where specific column is missing
df_dropped = df.dropna(subset=['column_name'])

# Remove rows with threshold of missing values
df_dropped = df.dropna(thresh=len(df)*0.5)  # keep rows with at least 50% data
```

### 2. Mean/Median/Mode Imputation

**Pros**: Simple, preserves data size
**Cons**: Reduces variance, ignores relationships

```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
df['column'] = imputer.fit_transform(df[['column']])

# Median imputation
imputer = SimpleImputer(strategy='median')
df['column'] = imputer.fit_transform(df[['column']])

# Mode imputation
imputer = SimpleImputer(strategy='most_frequent')
df['column'] = imputer.fit_transform(df[['column']])
```

### 3. Forward Fill / Backward Fill (Time Series)

```python
# Forward fill
df['column'] = df['column'].fillna(method='ffill')

# Backward fill
df['column'] = df['column'].fillna(method='bfill')

# Fill with constant value
df['column'] = df['column'].fillna(0)
```

### 4. Interpolation

```python
# Linear interpolation
df['column'] = df['column'].interpolate(method='linear')

# Polynomial interpolation
df['column'] = df['column'].interpolate(method='polynomial', order=2)

# Spline interpolation
df['column'] = df['column'].interpolate(method='spline', order=3)
```

### 5. K-Nearest Neighbors (KNN) Imputation

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

### 6. Multiple Imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df)
```

## Choosing an Imputation Strategy

1. **Understand the data**: Identify MCAR, MAR, or MNAR
2. **Data type**: Numerical vs categorical
3. **Percentage missing**: High percentage might need different approach
4. **Relationships**: Use methods that preserve correlations
5. **Domain knowledge**: Expert input on reasonable values

## Best Practices

- Always create missing value indicator columns before imputation
- Fit imputer on training data, apply to test data
- Consider multiple imputation for uncertainty estimation
- Validate results after imputation
- Document imputation strategy used

## Performance Comparison

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# Simulate original data
original = np.array([1, 2, 3, 4, 5])

# Introduce missing values
data_with_missing = np.array([1, np.nan, 3, np.nan, 5])

# Test different imputation methods
methods = {
    'Mean': mean_imputation(data_with_missing),
    'KNN': knn_imputation(data_with_missing),
    'Interpolation': interpolation(data_with_missing)
}

# Calculate error for each method
for method, result in methods.items():
    error = mean_squared_error(original, result)
    print(f"{method}: MSE = {error:.4f}")
```

## Summary

Choosing the right imputation method requires understanding your data, the nature of missing values, and the impact on downstream analysis. Test multiple approaches and validate results.
