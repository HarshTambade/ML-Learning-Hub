03_feature_scaling_normalization.md  # Feature Scaling and Normalization

## Introduction

Feature scaling is essential for machine learning models that are sensitive to feature magnitude, such as distance-based or gradient-descent algorithms.

## Why Scale Features?

1. **Improves Model Performance**: Many algorithms converge faster with scaled features
2. **Fair Feature Comparison**: Prevents high-magnitude features from dominating
3. **Regularization Effectiveness**: Scaling ensures regularization applies uniformly
4. **Distance Calculations**: Essential for KNN, K-Means, SVM
5. **Gradient Descent**: Convergence is faster with scaled data

## Scaling Techniques

### 1. Standardization (Z-score Normalization)

Scales features to mean=0, std=1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Formula: (x - mean) / std
```

**Pros**: Works well with normal distribution, symmetric scaling
**Cons**: Unbounded output

### 2. Min-Max Scaling (Normalization)

Scales features to range [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Formula: (x - min) / (max - min)
```

**Pros**: Bounded output, preserves shape
**Cons**: Sensitive to outliers

### 3. Robust Scaling

Uses median and IQR, resistant to outliers

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# Formula: (x - median) / IQR
```

### 4. Log Scaling

For skewed distributions

```python
import numpy as np

X_scaled = np.log(X + 1)  # +1 to handle zeros
```

### 5. Vector Normalization

Scales each sample to unit norm

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
X_scaled = normalizer.fit_transform(X_train)

# L2: Euclidean norm
# L1: Manhattan norm
```

## When to Use Which Method

| Algorithm | Recommended Scaling |
|-----------|-------------------|
| Linear Regression | StandardScaler |
| Logistic Regression | StandardScaler |
| KNN | MinMaxScaler or StandardScaler |
| K-Means | StandardScaler |
| SVM | StandardScaler |
| Tree-based | No scaling needed |
| Neural Networks | StandardScaler or MinMaxScaler |
| PCA | StandardScaler |

## Important Considerations

### Fit on Training Data Only

```python
# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics

# WRONG - Data leakage!
scaler_train = StandardScaler()
scaler_test = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_test_scaled = scaler_test.fit_transform(X_test)  # Different statistics!
```

### Handle Missing Values

```python
# Impute before scaling
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)
```

## Comparison Example

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
X = np.array([[1, 100], [2, 200], [3, 300]])

# StandardScaler
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print("StandardScaler:")
print(X_std)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
print("\nMinMaxScaler:")
print(X_minmax)
```

## Summary

Choose scaling method based on:
1. Algorithm requirements
2. Data distribution
3. Presence of outliers
4. Domain knowledge

Always fit on training data and apply to test data to prevent data leakage.
