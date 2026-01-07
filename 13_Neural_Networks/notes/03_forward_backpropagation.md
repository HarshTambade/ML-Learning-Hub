# Forward and Backpropagation

## Table of Contents
1. [Introduction](#introduction)
2. [Forward Propagation](#forward-propagation)
3. [Backpropagation](#backpropagation)
4. [Gradient Computation](#gradient-computation)
5. [Chain Rule](#chain-rule)
6. [Backprop Algorithm](#backprop-algorithm)
7. [Computational Complexity](#computational-complexity)
8. [Common Issues](#common-issues)
9. [Code Examples](#code-examples)
10. [Resources](#resources)

---

## Introduction

Forward propagation and backpropagation are the two fundamental algorithms in neural network training:

- **Forward Propagation**: Compute output from input
- **Backpropagation**: Compute gradients for weight updates

Together, they enable neural networks to learn from data.

---

## Forward Propagation

### Definition

Forward propagation (feedforward) computes the network output given an input by passing data through each layer sequentially.

### Algorithm

```
For each layer l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]     # Weighted sum
    a[l] = activate(z[l])            # Apply activation

Output: a[L]  (predictions)
```

### Simple Example

```
Network: 2 -> 3 -> 1 (2 inputs, 3 hidden, 1 output)

Input: x = [2, 1]
Weights Layer 1: W1 = [[0.5, 0.3], [0.2, 0.4], [0.1, 0.9]]
Biases Layer 1: b1 = [0.1, 0.2, 0.3]

z1 = W1 @ x + b1
   = [[0.5, 0.3],    [[2],    [[0.1],
      [0.2, 0.4],  @  [1]]  +  [0.2],
      [0.1, 0.9]]              [0.3]]
   = [[1.3], [1.0], [2.0]] + [[0.1], [0.2], [0.3]]
   = [[1.4], [1.2], [2.3]]

a1 = ReLU(z1) = [[1.4], [1.2], [2.3]]  # All positive, ReLU unchanged

Weights Layer 2: W2 = [[0.5, 0.6, 0.4]]
Biases Layer 2: b2 = [0.1]

z2 = W2 @ a1 + b2
   = [0.5, 0.6, 0.4] @ [[1.4], [1.2], [2.3]] + [0.1]
   = 0.7 + 0.72 + 0.92 + 0.1 = 2.44

a2 = Sigmoid(2.44) ≈ 0.92  # Final output
```

### Cost

For a network with layers n1, n2, ..., nL:

```
Forward cost ≈ 2 * (n1*n2 + n2*n3 + ... + n(L-1)*nL) multiplications
```

---

## Backpropagation

### Overview

Backpropagation computes how much each weight contributes to the output error, enabling weight updates.

### Key Insight

The output depends on ALL weights through nested function composition:

```
Output = f(W3 @ f(W2 @ f(W1 @ input + b1) + b2) + b3)
```

Chain rule allows efficient gradient computation!

### Backprop Algorithm

```
# 1. Forward pass (compute all activations)
For each layer l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activate(z[l])

# 2. Compute output error
dL/da[L] = predicted - target

# 3. Backward pass (compute gradients)
For each layer l = L down to 1:
    dL/dz[l] = (dL/da[l]) * activate'(z[l])
    dL/dW[l] = dL/dz[l] @ a[l-1]^T
    dL/db[l] = sum(dL/dz[l])
    dL/da[l-1] = W[l]^T @ dL/dz[l]

# 4. Update weights
For each layer l = 1 to L:
    W[l] = W[l] - learning_rate * dL/dW[l]
    b[l] = b[l] - learning_rate * dL/db[l]
```

---

## Gradient Computation

### Loss Function Gradient

For MSE loss:
```
L = (1/2m) * Σ(y_pred - y_true)²

dL/dy_pred = (y_pred - y_true) / m
```

For Cross-Entropy:
```
L = -Σ(y_true * log(y_pred))

dL/dy_pred = -y_true / y_pred
```

### Activation Derivatives

**Sigmoid**:
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x) * (1 - σ(x))
```

**ReLU**:
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0
```

**Tanh**:
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

---

## Chain Rule

### Concept

For composite functions:
```
y = f(g(x))
dy/dx = (dy/dg) * (dg/dx)
```

### Multilayer Application

For network output through multiple layers:
```
L = loss(output) = loss(a3)
where a3 = activate(W3 @ a2 + b3)
where a2 = activate(W2 @ a1 + b2)
where a1 = activate(W1 @ input + b1)

dL/dW1 = (dL/da3) * (da3/dz3) * (dz3/da2) * (da2/dz2) * (dz2/da1) * (da1/dz1) * (dz1/dW1)
```

Backpropagation efficiently computes this product!

---

## Backprop Algorithm

### Step-by-Step

**Step 1: Forward Pass**
```python
for l in range(1, L+1):
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activation(z[l])
```

**Step 2: Compute Output Error**
```python
delta[L] = (a[L] - y) * activation_derivative(z[L])
# For softmax + cross-entropy: delta[L] = a[L] - y
```

**Step 3: Backward Pass**
```python
for l in range(L-1, 0, -1):
    dW[l] = delta[l] @ a[l-1].T / m
    db[l] = sum(delta[l]) / m
    delta[l-1] = (W[l].T @ delta[l]) * activation_derivative(z[l-1])
```

**Step 4: Update Parameters**
```python
for l in range(1, L+1):
    W[l] -= learning_rate * dW[l]
    b[l] -= learning_rate * db[l]
```

---

## Computational Complexity

### Forward Pass
- **Time**: O(n_layers * avg_neurons²)
- **Space**: O(all_activations)

### Backward Pass
- **Time**: ~2x Forward pass
- **Space**: O(all_activations) + O(all_gradients)

### Total Training Step
```
Time ≈ 3 * Forward Pass
```

---

## Common Issues

### 1. Vanishing Gradient

**Problem**: Gradients become very small in deep networks

```
Gradient = product of activation derivatives
If each derivative < 1: product → 0
```

**Solutions**:
- Use ReLU (no saturation)
- Batch normalization
- Residual connections
- Careful weight initialization

### 2. Exploding Gradient

**Problem**: Gradients become very large

```
If each derivative > 1: product → ∞
Weights update too much
Training becomes unstable
```

**Solutions**:
- Gradient clipping
- Batch normalization
- Smaller learning rate
- Weight regularization

### 3. Dead ReLU

**Problem**: ReLU neurons output 0, gradients become 0

```
If z < 0 always: output = 0, gradient = 0
Neuron never updates
```

**Solutions**:
- Leaky ReLU
- ELU
- Careful initialization

---

## Code Examples

### Simple Backpropagation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Forward pass
z = W @ x + b
a = sigmoid(z)
output = a

# Backward pass
error = output - y
dL_dz = error * sigmoid_derivative(z)
dL_dW = dL_dz @ x.T / m
dL_db = np.mean(dL_dz)

# Update
W -= learning_rate * dL_dW
b -= learning_rate * dL_db
```

### With TensorFlow

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Backprop happens inside fit()
model.fit(X_train, y_train, epochs=10)
```

---

## Resources

### Key Papers
- "Learning Representations by Back-propagating Errors" - Rumelhart, Hinton, Williams (1986)
- "Gradient-based Learning Applied to Document Recognition" - LeCun et al. (1998)

### Tutorials
- 3Blue1Brown: Backpropagation Calculus
- CS231n: Backpropagation and Neural Networks
- DeepLearning.AI: How Backprop Works

### Interactive Tools
- TensorFlow Playground
- Neural Network 3D Visualization
- GradientFlow Visualization

---

**Last Updated**: January 2026
**Difficulty Level**: Intermediate to Advanced
**Prerequisites**: Calculus, Linear Algebra, Neural Network Basics
