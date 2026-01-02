# SVM Troubleshooting and Tips

## Common Issues and Solutions

### Issue 1: Poor Model Performance

**Symptoms**:
- Low accuracy on both training and test data
- Model not learning from data

**Causes**:
1. Features not scaled
2. Wrong kernel selected
3. Poor hyperparameter values
4. Data quality issues

**Solutions**:
```python
# Always scale features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try different kernels
for kernel in ['linear', 'rbf', 'poly']:
    svc = SVC(kernel=kernel)
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print(f"{kernel}: {score}")

# Use GridSearchCV for tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Issue 2: Overfitting

**Symptoms**:
- High training accuracy but low test accuracy
- Large gap between train/test performance

**Causes**:
- C parameter too large
- Gamma parameter too large (RBF)
- Too complex polynomial degree
- Insufficient training data

**Solutions**:
```python
# Reduce C (regularization)
svc = SVC(C=0.1, kernel='rbf')  # Smaller C = larger margin

# Reduce gamma (RBF)
svc = SVC(C=1, gamma=0.001, kernel='rbf')  # Smaller gamma = smoother

# Use lower polynomial degree
svc = SVC(kernel='poly', degree=2)  # degree=2 instead of 5

# Increase training data
# Collect more samples

# Use cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svc, X, y, cv=5)
print(f"CV scores: {scores}")
```

### Issue 3: Underfitting

**Symptoms**:
- Low accuracy on both training and test data
- Model too simple

**Causes**:
- C parameter too small
- Gamma parameter too small (RBF)
- Using linear kernel for non-linear data
- Too few features

**Solutions**:
```python
# Increase C (less regularization)
svc = SVC(C=100, kernel='rbf')  # Larger C = smaller margin

# Increase gamma (RBF)
svc = SVC(C=1, gamma=1, kernel='rbf')  # Larger gamma = wiggly

# Use RBF kernel instead of linear
svc = SVC(kernel='rbf')

# Create new features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

### Issue 4: Slow Training

**Symptoms**:
- Training takes too long
- Memory usage is high

**Causes**:
- Dataset too large (> 100K samples)
- Complex kernel computations
- Using RBF on large data

**Solutions**:
```python
# Use linear kernel for large datasets
svc = SVC(kernel='linear')  # O(n) instead of O(n^2)

# Sample data for initial exploration
from sklearn.utils import shuffle
X_sample = shuffle(X)[:10000]  # Use subset
y_sample = shuffle(y)[:10000]

# Use SGDClassifier for very large data
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge')
sgd.fit(X, y)
```

### Issue 5: Feature Scaling Problems

**Symptoms**:
- Model performance varies greatly
- Some features dominate
- Numerical instability

**Solutions**:
```python
# Always scale before SVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max scaling (0 to 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# For sparse data
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```

## Performance Optimization Tips

### Tip 1: Choose Right Kernel

```python
# Linear data or high dimensions
svc_linear = SVC(kernel='linear', C=1)

# Non-linear data, binary classification
svc_rbf = SVC(kernel='rbf', C=1, gamma=0.1)

# Polynomial relationships
svc_poly = SVC(kernel='poly', degree=2, C=1)
```

### Tip 2: Efficient Hyperparameter Search

```python
# Coarse search first
coarse_grid = {'C': [0.01, 0.1, 1, 10, 100]}
coarse_search = GridSearchCV(SVC(), coarse_grid, cv=3)
coarse_search.fit(X, y)

# Fine search around best params
fine_grid = {'C': [0.5, 1, 2, 5]}
fine_search = GridSearchCV(SVC(), fine_grid, cv=5)
fine_search.fit(X, y)
```

### Tip 3: Handle Class Imbalance

```python
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# Automatic balancing
svc = SVC(class_weight='balanced')

# Manual weights
weights = compute_class_weight('balanced', np.unique(y), y)
class_weight_dict = {i: w for i, w in enumerate(weights)}
svc = SVC(class_weight=class_weight_dict)
```

### Tip 4: Get Probability Estimates

```python
# Enable probability for soft predictions
svc = SVC(probability=True)
svc.fit(X_train, y_train)

# Get probabilities
proba = svc.predict_proba(X_test)
print(f"Probability of class 1: {proba[:, 1]}")
```

### Tip 5: Interpret SVM Decisions

```python
# Number of support vectors
n_support = svc.n_support_
print(f"Support vectors per class: {n_support}")

# Feature importance (for linear SVM only)
importances = np.abs(svc.coef_[0])
print(f"Important features: {np.argsort(importances)[-5:]}")
```

## Best Practices Checklist

- [ ] **Scale Features**: Always use StandardScaler
- [ ] **Train-Test Split**: Keep test data separate for final evaluation
- [ ] **Cross-Validation**: Use 5-fold or 10-fold CV
- [ ] **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
- [ ] **Handle Class Imbalance**: Use class_weight='balanced'
- [ ] **Start Simple**: Begin with linear kernel, then try RBF
- [ ] **Validate Results**: Check with multiple metrics (accuracy, F1, AUC)
- [ ] **Save Model**: Use joblib to save trained models
- [ ] **Monitor Performance**: Track train/test scores to detect overfitting
- [ ] **Document Parameters**: Record best hyperparameters

## Code Template

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
svc = SVC()
grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_train, y_train)

# 5. Evaluate
print(f"Best params: {grid.best_params_}")
print(f"CV score: {grid.best_score_:.3f}")
y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(grid.best_estimator_, 'svm_model.pkl')
```
