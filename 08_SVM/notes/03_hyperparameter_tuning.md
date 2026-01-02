# Hyperparameter Tuning for SVM

## Critical Hyperparameters

SVM has several key hyperparameters that significantly impact model performance:

### 1. C (Regularization Parameter)

**Range**: Typically 0.1 to 1000 (log scale)

**Effect**:
- **Small C** (e.g., 0.01): Large margin, more tolerance for errors
  - Simpler decision boundary
  - Better generalization
  - Underfitting risk
  - More support vectors

- **Large C** (e.g., 100): Small margin, less tolerance for errors
  - Complex decision boundary
  - Better training accuracy
  - Overfitting risk
  - Fewer support vectors

**Visual Interpretation**:
```
Small C:  O___O___O  (smooth, wide margin)
Large C:  O-O-O-O    (tight, zigzag boundary)
```

### 2. Gamma (γ) - RBF Kernel Only

**Range**: 0.0001 to 10 (log scale)

**Effect** (controls RBF influence):
- **Small γ** (e.g., 0.001): Wide influence of each training point
  - Smooth decision boundary
  - Simple model
  - Underfitting risk
  - Training takes longer

- **Large γ** (e.g., 100): Narrow influence of each training point
  - Wiggly decision boundary
  - Complex model
  - Overfitting risk
  - Training is faster

**Formula**: RBF kernel = exp(-γ·||x_i - x_j||²)
- Small γ: All points contribute significantly
- Large γ: Only nearby points matter

### 3. Kernel Type

```
Linear:     'linear'   - Fast, good for high dimensions
Polynomial: 'poly'     - Captures polynomial relationships
RBF:        'rbf'      - Most versatile, handles complex patterns
Sigmoid:    'sigmoid'  - Similar to neural networks
```

### 4. Degree (Polynomial Kernel)

**Range**: 2, 3, 4, ... (usually not > 5)

**Effect**:
- Higher degree = more complex polynomial relationships
- Too high: overfitting and slow training
- Usually degree 2-3 works best

## Tuning Strategies

### Strategy 1: Grid Search

Test all combinations of hyperparameters from predefined grids.

**Python Example**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

svc = SVC()
grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

**Advantages**:
- Exhaustive search
- Guaranteed to find best combination
- Parallelizable

**Disadvantages**:
- Computationally expensive
- Exponential growth with parameters
- O(n^k) complexity for k parameters

### Strategy 2: Random Search

Test random combinations from parameter distributions.

**Python Example**:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(0.001, 10),
    'kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(
    SVC(), param_dist, n_iter=20, cv=5, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

**Advantages**:
- Much faster than grid search
- Can discover unexpected good combinations
- Better for continuous parameters

**Disadvantages**:
- May miss optimal combination
- Less thorough

### Strategy 3: Bayesian Optimization

Uses probabilistic model to guide search.

**Python Example**:
```python
from skopt import gp_minimize

def objective(params):
    C, gamma = params
    svc = SVC(C=C, gamma=gamma, kernel='rbf')
    score = cross_val_score(svc, X_train, y_train, cv=5).mean()
    return -score

result = gp_minimize(
    objective,
    dimensions=[(0.1, 100), (0.001, 10)],
    n_calls=30,
    random_state=42
)
```

**Advantages**:
- Efficient search
- Learns from previous iterations
- Good for expensive evaluations

**Disadvantages**:
- More complex
- Requires additional library

## Practical Tuning Process

### Step 1: Coarse Grid Search
Start with a rough grid to narrow down parameter ranges.

```python
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1],
}
```

### Step 2: Fine Grid Search
Refine around the best parameters.

```python
param_grid = {
    'C': [0.5, 1, 2, 5, 10],
    'gamma': [0.005, 0.01, 0.02, 0.05],
}
```

### Step 3: Kernel Selection
Compare different kernels on best C and gamma.

```python
for kernel in ['linear', 'poly', 'rbf']:
    svc = SVC(C=best_C, gamma=best_gamma, kernel=kernel)
    score = cross_val_score(svc, X_train, y_train, cv=5).mean()
    print(f"{kernel}: {score}")
```

## Recommended Parameter Ranges

### For RBF Kernel
```
C:     [0.1, 1, 10, 100, 1000]
gamma: [0.0001, 0.001, 0.01, 0.1, 1]
```

### For Linear Kernel
```
C:     [0.01, 0.1, 1, 10, 100]
(gamma not used)
```

### For Polynomial Kernel
```
C:     [0.1, 1, 10, 100]
degree: [2, 3, 4]
gamma:  [0.001, 0.01, 0.1]
```

## Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

# Use StratifiedKFold for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    SVC(), param_grid, 
    cv=skf,
    scoring='f1_weighted',  # Use appropriate metric
    n_jobs=-1
)
```

## Common Mistakes to Avoid

1. **Not scaling features**: Always normalize before tuning
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   ```

2. **Tuning on test set**: Use only training data
   - Use cross-validation on training set
   - Final evaluation on untouched test set

3. **Ignoring class imbalance**: Use class_weight='balanced'
   ```python
   svc = SVC(class_weight='balanced')
   ```

4. **Too wide parameter ranges**: Start coarse, then refine

5. **Not enough folds**: Use at least 5-fold CV

## Evaluation Metrics

Choose metric based on problem:
```python
scoring = 'accuracy'        # Balanced classes
scoring = 'f1_weighted'     # Imbalanced classes
scoring = 'precision'       # Minimize false positives
scoring = 'recall'          # Minimize false negatives
scoring = 'roc_auc'         # General classification
```

## Complete Example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

# Evaluation
best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(f"Best params: {grid.best_params_}")
```
