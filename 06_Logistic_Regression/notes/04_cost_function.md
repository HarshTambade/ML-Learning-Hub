# Cost Function in Logistic Regression

## Overview

The cost function (also called loss function) measures how well the logistic regression model performs. It quantifies the difference between predicted probabilities and actual labels, guiding the optimization process during training.

## Binary Cross-Entropy Loss

### Definition

For binary classification, logistic regression uses **Binary Cross-Entropy (BCE)** loss:

```
J(w, b) = -1/m * Σ[y_i * log(h(x_i)) + (1 - y_i) * log(1 - h(x_i))]
```

Where:
- `m` = number of training samples
- `y_i` = true label (0 or 1)
- `h(x_i)` = predicted probability (0 to 1)
- `log` = natural logarithm

### Why Cross-Entropy?

1. **Convex Function**: Has a single global minimum
2. **Probabilistic Interpretation**: Aligns with likelihood estimation
3. **Penalty for Wrong Predictions**: Penalizes confident wrong predictions heavily
4. **Gradient Properties**: Provides good gradients for optimization

## Mathematical Intuition

### Breaking Down the Formula

```
Cost = -[y * log(h) + (1-y) * log(1-h)]
```

**Case 1: When y = 1 (True Label)**
```
Cost = -log(h)

If h → 1:  Cost → 0  (correct prediction, low cost)
If h → 0:  Cost → ∞  (wrong prediction, high cost)
```

**Case 2: When y = 0 (True Label)**
```
Cost = -log(1 - h)

If h → 0:  Cost → 0  (correct prediction, low cost)
If h → 1:  Cost → ∞  (wrong prediction, high cost)
```

## Implementation in Python

### Manual Implementation

```python
import numpy as np

def compute_cost(X, y, w, b):
    """
    Compute binary cross-entropy cost
    
    Parameters:
    X : array of shape (m, n) - training features
    y : array of shape (m,) - true labels
    w : array of shape (n,) - weights
    b : scalar - bias
    
    Returns:
    cost : scalar - average cost
    """
    m = X.shape[0]
    
    # Compute predictions
    z = np.dot(X, w) + b
    h = 1 / (1 + np.exp(-z))  # sigmoid
    
    # Avoid log(0) by clipping predictions
    h = np.clip(h, 1e-7, 1 - 1e-7)
    
    # Compute cost
    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    return cost

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
w = np.array([0.1, 0.2])
b = 0.5

cost = compute_cost(X_train, y_train, w, b)
print(f"Cost: {cost:.4f}")
```

### Using scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X, y)

# Get log loss (cost)
log_loss = -np.mean(y * np.log(model.predict_proba(X)[:, 1]) + 
                     (1 - y) * np.log(1 - model.predict_proba(X)[:, 1]))
print(f"Log Loss: {log_loss:.4f}")
```

## Visualization of Cost Function

### Cost vs Prediction Probability

```python
import matplotlib.pyplot as plt
import numpy as np

# When true label = 1
h = np.linspace(0.001, 0.999, 100)
cost_y1 = -np.log(h)

# When true label = 0
cost_y0 = -np.log(1 - h)

plt.figure(figsize=(10, 6))
plt.plot(h, cost_y1, label='y = 1 (true label)', linewidth=2)
plt.plot(h, cost_y0, label='y = 0 (true label)', linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Cost')
plt.title('Binary Cross-Entropy Cost Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 5)
plt.show()
```

## Cost Function Behavior

### Property 1: Convexity
- The cost function is **strictly convex**
- Guarantees unique global minimum
- Any local minimum is the global minimum

### Property 2: Asymmetry
- Different penalties for different errors
- Penalizes confident wrong predictions heavily
- Encourages well-calibrated probabilities

### Property 3: Smoothness
- Continuous and differentiable everywhere
- Enables gradient-based optimization
- Provides direction for parameter updates

## Gradient of Cost Function

### Partial Derivatives

```
∂J/∂w_j = 1/m * Σ (h(x_i) - y_i) * x_ij
∂J/∂b = 1/m * Σ (h(x_i) - y_i)
```

Where:
- `h(x_i)` = predicted probability
- `y_i` = true label
- `x_ij` = j-th feature of sample i

### Implementation

```python
def compute_gradients(X, y, h):
    """
    Compute gradients of cost function
    """
    m = X.shape[0]
    error = h - y
    
    dw = 1/m * np.dot(X.T, error)
    db = 1/m * np.sum(error)
    
    return dw, db
```

## Regularization

### L2 Regularization (Ridge)

```
J(w, b) = BCE_loss + (λ/2m) * Σ w_j²
```

**Purpose**: Prevents overfitting by penalizing large weights

```python
from sklearn.linear_model import LogisticRegression

# L2 regularization (default)
model = LogisticRegression(C=1.0, penalty='l2')
model.fit(X, y)
```

### L1 Regularization (Lasso)

```
J(w, b) = BCE_loss + (λ/m) * Σ |w_j|
```

**Purpose**: Feature selection via sparsity

```python
from sklearn.linear_model import LogisticRegression

# L1 regularization
model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear')
model.fit(X, y)
```

## Cost Function Interpretation

### Perfect Model
- All predictions correct (h = y for all samples)
- Cost = 0
- Not achievable in practice

### Good Model
- Cost < 0.5
- Most predictions are confident and correct
- Reasonable fit to data

### Poor Model
- Cost > 0.5
- Many incorrect or uncertain predictions
- Needs improvement

### Completely Wrong Model
- Cost → ∞
- Confident wrong predictions
- Model is learning incorrect patterns

## Practical Considerations

### 1. **Numerical Stability**
```python
# Avoid log(0) by clipping
h = np.clip(h, 1e-7, 1 - 1e-7)
```

### 2. **Regularization Parameter**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Grid search for best C
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(LogisticRegression(), params, cv=5)
model.fit(X, y)
```

### 3. **Class Imbalance**
```python
from sklearn.linear_model import LogisticRegression

# Balanced class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)
```

## Comparison with MSE

| Aspect | Cross-Entropy | Mean Squared Error |
|--------|---------------|-------------------|
| Convexity | Strictly convex | Convex but may have flat regions |
| Penalties | Exponential for wrong predictions | Quadratic |
| Interpretation | Probabilistic | Geometric distance |
| Gradient | Works well with sigmoid | Can be small with sigmoid |
| Recommended | ✅ Yes for classification | ❌ No for classification |

## Real-World Example

### Email Spam Detection

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Compute cost
from sklearn.metrics import log_loss

cost_train = log_loss(y_train, model.predict_proba(X_train))
cost_test = log_loss(y_test, model.predict_proba(X_test))

print(f"Training Cost: {cost_train:.4f}")
print(f"Testing Cost: {cost_test:.4f}")
```

## Key Takeaways

✅ **Binary Cross-Entropy** is the standard cost function for logistic regression

✅ **Convex function** ensures unique global minimum

✅ **Heavily penalizes** confident wrong predictions

✅ **Differentiable** for gradient-based optimization

✅ **Regularization** prevents overfitting

✅ **Numerical stability** requires clipping predictions

## YouTube Resources

### Recommended Videos:

1. **"Cost Function for Logistic Regression" - StatQuest**
   - Clear explanation of cross-entropy loss
   - Intuitive examples
   - [Watch on YouTube](https://www.youtube.com/watch?v=BfKanl1aSG0)

2. **"Binary Cross-Entropy Loss" - Andrew Ng (Coursera)**
   - Mathematical derivation
   - Intuition behind the formula
   - [Playlist](https://www.youtube.com/watch?v=TZyzIM9-rBs)

3. **"Loss Functions Explained" - Jeremy Jordan**
   - Comparison of different loss functions
   - When to use which
   - [YouTube Tutorial](https://www.youtube.com/watch?v=kFr0JZ7X2F8)

4. **"Cost Function Optimization" - Krish Naik**
   - Practical implementation
   - Gradient computation
   - [Complete Tutorial](https://www.youtube.com/watch?v=6OrYWZAP7Fs)

5. **"Log Loss and Regularization" - Victor Lavrenko**
   - Detailed mathematical background
   - L1 and L2 regularization
   - [YouTube](https://www.youtube.com/watch?v=RIyP8jJ6Jb8)

## Further Reading

- James et al. "An Introduction to Statistical Learning" - Chapter 4.2
- Bishop "Pattern Recognition and Machine Learning" - Chapter 4.3
- Hastie, Tibshirani & Friedman "The Elements of Statistical Learning" - Chapter 4

---

**Last Updated**: 2024
**Difficulty Level**: Intermediate
**Prerequisites**: Logistic regression basics, Sigmoid function, Calculus basics
