# Gradient Descent Optimization in Logistic Regression

## Overview

Gradient descent is an iterative optimization algorithm that finds the parameters (weights and bias) that minimize the cost function. It's the foundation of how machine learning models learn from data.

## How Gradient Descent Works

### The Intuition

Imagine you're at the top of a mountain in the fog, and you want to reach the valley (minimum cost). You can't see the full landscape, but you can feel the slope beneath your feet. Gradient descent:

1. Checks the slope (gradient) in all directions
2. Steps downhill in the steepest direction
3. Repeats until reaching the valley

### Mathematical Foundation

**Update Rule**:
```
w_new = w_old - α * ∂J/∂w
b_new = b_old - α * ∂J/∂b
```

Where:
- `α` (alpha) = learning rate (step size)
- `∂J/∂w` = gradient of cost function with respect to weights
- `∂J/∂b` = gradient of cost function with respect to bias

## Types of Gradient Descent

### 1. Batch Gradient Descent (BGD)

**Uses entire dataset to compute gradient in each iteration**

```python
import numpy as np

def batch_gradient_descent(X, y, w, b, learning_rate, iterations):
    """
    Batch Gradient Descent for Logistic Regression
    """
    m = X.shape[0]
    cost_history = []
    
    for i in range(iterations):
        # Forward propagation
        z = np.dot(X, w) + b
        h = 1 / (1 + np.exp(-z))  # sigmoid
        
        # Compute gradients
        error = h - y
        dw = 1/m * np.dot(X.T, error)
        db = 1/m * np.sum(error)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Compute and store cost
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)
    
    return w, b, cost_history
```

**Pros**:
- Stable convergence
- Exact gradients
- Good for convex functions

**Cons**:
- Slow for large datasets
- Cannot update until entire batch is processed

### 2. Stochastic Gradient Descent (SGD)

**Uses one sample at a time to compute gradient**

```python
def stochastic_gradient_descent(X, y, w, b, learning_rate, iterations):
    """
    Stochastic Gradient Descent for Logistic Regression
    """
    m = X.shape[0]
    cost_history = []
    
    for epoch in range(iterations):
        for i in range(m):
            # Single sample
            x_i = X[i].reshape(1, -1)
            y_i = y[i]
            
            # Forward propagation
            z = np.dot(x_i, w) + b
            h = 1 / (1 + np.exp(-z))
            
            # Compute gradients for single sample
            error = h - y_i
            dw = x_i.T * error
            db = error
            
            # Update parameters
            w = w - learning_rate * dw
            b = b - learning_rate * db
        
        # Compute cost for entire batch
        z = np.dot(X, w) + b
        h = 1 / (1 + np.exp(-z))
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)
    
    return w, b, cost_history
```

**Pros**:
- Fast updates
- Can escape local minima
- Memory efficient

**Cons**:
- Noisy gradient estimates
- Unstable convergence
- May overshoot minimum

### 3. Mini-Batch Gradient Descent

**Uses small batches of samples**

```python
def mini_batch_gradient_descent(X, y, w, b, learning_rate, iterations, batch_size=32):
    """
    Mini-Batch Gradient Descent for Logistic Regression
    """
    m = X.shape[0]
    cost_history = []
    
    for epoch in range(iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            batch_m = X_batch.shape[0]
            
            # Forward propagation
            z = np.dot(X_batch, w) + b
            h = 1 / (1 + np.exp(-z))
            
            # Compute gradients
            error = h - y_batch
            dw = 1/batch_m * np.dot(X_batch.T, error)
            db = 1/batch_m * np.sum(error)
            
            # Update parameters
            w = w - learning_rate * dw
            b = b - learning_rate * db
        
        # Compute cost
        z = np.dot(X, w) + b
        h = 1 / (1 + np.exp(-z))
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)
    
    return w, b, cost_history
```

**Best of both worlds**:
- Reasonable updates (not too noisy, not too slow)
- Stable convergence
- Memory efficient
- Used in practice

## Learning Rate Effects

### Learning Rate Too Small
```
α = 0.0001

Result: Very slow convergence (many iterations needed)
```

### Learning Rate Just Right
```
α = 0.01

Result: Smooth, steady convergence to minimum
```

### Learning Rate Too Large
```
α = 1.0

Result: Overshooting, divergence, or oscillation
```

## Visualizing Gradient Descent

```python
import matplotlib.pyplot as plt

# After training, plot cost function
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.grid(True, alpha=0.3)
plt.show()
```

## Convergence Criteria

### 1. Fixed Number of Iterations
```python
for iteration in range(num_iterations):
    # update weights
```

### 2. Gradient Magnitude Threshold
```python
while np.max(np.abs(gradient)) > threshold:
    # update weights
```

### 3. Cost Change Threshold
```python
while abs(cost_new - cost_old) > threshold:
    # update weights
```

## Practical Implementation with scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5,
                          n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Batch Gradient Descent
model_bgd = LogisticRegression(solver='lbfgs', max_iter=1000)
model_bgd.fit(X_train, y_train)
print(f"BGD Score: {model_bgd.score(X_test, y_test):.4f}")

# Stochastic Gradient Descent
model_sgd = LogisticRegression(solver='saga', max_iter=1000)
model_sgd.fit(X_train, y_train)
print(f"SGD Score: {model_sgd.score(X_test, y_test):.4f}")
```

## Common Issues and Solutions

### Issue 1: Convergence Too Slow
**Solutions**:
- Increase learning rate (carefully)
- Use mini-batch instead of full batch
- Normalize/standardize features
- Use advanced optimizers (Adam, RMSprop)

### Issue 2: Overshooting (Loss Oscillates)
**Solutions**:
- Decrease learning rate
- Use learning rate decay
- Use momentum

### Issue 3: Getting Stuck in Local Minima
**Solutions**:
- Use different random initialization
- Increase learning rate
- Use stochastic variants

## Advanced Optimization Techniques

### Momentum

Adds previous update direction to current update:

```python
velocity = 0
beta = 0.9  # momentum coefficient

for iteration in range(num_iterations):
    gradient = compute_gradient(X, y, w, b)
    velocity = beta * velocity - learning_rate * gradient
    w = w + velocity
```

### Adam Optimizer

Combines momentum and adaptive learning rates:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', optimizer='adam', learning_rate='adaptive')
model.fit(X_train, y_train)
```

## Step-by-Step Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize parameters
w = np.zeros(X.shape[1])
b = 0
alpha = 0.1
iterations = 100

# Gradient Descent
for i in range(iterations):
    # Forward pass
    z = np.dot(X, w) + b
    h = 1 / (1 + np.exp(-z))
    
    # Compute cost
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    # Backward pass
    dw = np.dot(X.T, (h - y)) / X.shape[0]
    db = np.mean(h - y)
    
    # Update
    w -= alpha * dw
    b -= alpha * db
    
    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1}, Cost: {cost:.4f}")
```

## Key Takeaways

✅ **Gradient descent** finds optimal parameters by following negative gradients

✅ **Learning rate** controls step size - critical hyperparameter

✅ **Batch GD** is stable but slow for large datasets

✅ **SGD** is fast but noisy; mini-batch is the sweet spot

✅ **Convergence** depends on learning rate, initialization, and data

✅ **Feature scaling** significantly improves convergence

✅ **Advanced optimizers** (Adam, RMSprop) handle learning rate automatically

## YouTube Resources

### Recommended Videos:

1. **"Gradient Descent Explained" - StatQuest**
   - Visual explanation with animations
   - Cost function surface visualization
   - [Watch on YouTube](https://www.youtube.com/watch?v=sDv4f4s2SB8)

2. **"Gradient Descent" - Andrew Ng (Coursera)**
   - Mathematical foundation
   - Learning rate effect visualization
   - [Playlist](https://www.youtube.com/watch?v=R9lkBUyCaGQ)

3. **"Stochastic Gradient Descent" - Jeremy Jordan**
   - SGD vs BGD comparison
   - Mini-batch strategies
   - [YouTube Tutorial](https://www.youtube.com/watch?v=9ZfDVG2Sh4c)

4. **"Optimization Algorithms" - Krish Naik**
   - Momentum, Adam, RMSprop
   - Implementation details
   - [Complete Tutorial](https://www.youtube.com/watch?v=HvZ_CbdnhaY)

5. **"Advanced Optimizers" - Yann LeCun (Slides)**
   - Adam, AdaGrad, RMSprop comparison
   - Theoretical analysis
   - [YouTube Lecture](https://www.youtube.com/watch?v=H7suYHKfkx4)

## Further Reading

- Goodfellow, Bengio & Courville "Deep Learning" - Chapter 8
- Boyd & Vandenberghe "Convex Optimization"
- Ruder "An overview of gradient descent optimization algorithms"

---

**Last Updated**: 2024
**Difficulty Level**: Intermediate
**Prerequisites**: Cost function understanding, Calculus (derivatives), Linear Algebra basics
