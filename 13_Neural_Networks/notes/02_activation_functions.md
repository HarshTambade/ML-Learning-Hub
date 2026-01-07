# Activation Functions in Neural Networks

## Table of Contents
1. [Introduction](#introduction)
2. [Why Activation Functions?](#why-activation-functions)
3. [Sigmoid Function](#sigmoid-function)
4. [Tanh Function](#tanh-function)
5. [ReLU (Rectified Linear Unit)](#relu)
6. [Leaky ReLU](#leaky-relu)
7. [ELU (Exponential Linear Unit)](#elu)
8. [Softmax Function](#softmax)
9. [Comparison & Selection](#comparison)
10. [Resources](#resources)

---

## Introduction

Activation functions are mathematical functions applied to neurons' output. They introduce non-linearity to neural networks, enabling them to learn complex patterns.

### Why Are They Important?

- **Non-linearity**: Enable networks to learn non-linear relationships
- **Gradient Flow**: Affect backpropagation effectiveness
- **Computational Efficiency**: Impact training speed
- **Output Range**: Determine output constraints

---

## Why Activation Functions?

### Problem Without Activation Functions

Without activation functions (linear networks):

```
y = W3 * (W2 * (W1 * x + b1) + b2) + b3
y = (W3 * W2 * W1) * x + (W3 * W2 * b1 + W3 * b2 + b3)
y = W * x + b  (still linear)
```

No matter how many layers, output is still linear!

### With Activation Functions

```
a1 = f(W1 * x + b1)
a2 = f(W2 * a1 + b2)
y = f(W3 * a2 + b3)
```

Now: Non-linear, can approximate any function!

---

## Sigmoid Function

### Formula

```
σ(x) = 1 / (1 + e^(-x))
```

### Characteristics

- **Range**: (0, 1)
- **Derivative**: σ'(x) = σ(x) * (1 - σ(x))
- **Shape**: S-shaped curve
- **Smooth**: Differentiable everywhere

### Graph

```
     1.0 |       ****
     0.5 |      **
       0 |***
    -0.5 |
       -5 -2.5 0  2.5  5
```

### Pros
- Output probability-like [0,1]
- Historically popular
- Smooth gradient

### Cons
- **Vanishing Gradient**: Gradient near 0 or 1 is very small
- **Not Zero-Centered**: Output always positive
- **Slow Convergence**: Due to gradient issues

### Usage
- **Binary Classification**: Output layer
- **Rarely**: Hidden layers (replaced by ReLU)

### Code Example

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Example
x = np.array([0, 1, -1, 2])
print(sigmoid(x))  # [0.5, 0.731, 0.269, 0.881]
```

---

## Tanh Function

### Formula

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x) = 2*sigmoid(2x) - 1
```

### Characteristics

- **Range**: (-1, 1)
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- **Zero-Centered**: Output centered at 0
- **Stronger Gradients**: Derivative range [0, 1] vs sigmoid [0, 0.25]

### Graph

```
       1 |      ***
       0 |-----***-----
      -1 |***
       -5 -2.5 0  2.5  5
```

### Pros
- Zero-centered output
- Stronger gradients than sigmoid
- Better convergence

### Cons
- Still suffers from vanishing gradients
- Slower than ReLU

### Usage
- Hidden layers (better than sigmoid)
- RNNs and LSTMs
- When output needs negative values

### Code Example

```python
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2

# Example
x = np.array([0, 1, -1, 2])
print(tanh(x))  # [0, 0.762, -0.762, 0.964]
```

---

## ReLU (Rectified Linear Unit)

### Formula

```
ReLU(x) = max(0, x)
```

### Characteristics

- **Range**: [0, ∞)
- **Derivative**: 0 if x < 0, 1 if x > 0, undefined at x=0
- **Simple**: Computationally efficient
- **Sparse Activation**: Only some neurons active

### Graph

```
       4 |      /
       2 |     /
       0 |----/
      -2 |--------
     -5  -2.5 0  2.5  5
```

### Pros
- **Fast Convergence**: Simple, efficient
- **No Vanishing Gradient**: For positive x
- **Sparse**: Natural feature selection
- **Biologically Inspired**: Mimics neurons

### Cons
- **Dying ReLU**: Neurons stuck at 0
- **Not Zero-Centered**: Output non-negative
- **Non-Differentiable**: At x=0 (ignored in practice)

### Dying ReLU Problem

```
If weights become too negative:
  x * weight + bias < 0 for all inputs
  ReLU output = 0 always
  Gradient = 0
  Weights never update
  Neuron "dead"
```

### Usage
- **Most Popular**: Hidden layers (standard)
- **CNNs, MLPs**: Default choice
- **Not for Output**: Use sigmoid/softmax for classification

### Code Example

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Example
x = np.array([1, -2, 0, 3, -1])
print(relu(x))  # [1, 0, 0, 3, 0]
print(relu_derivative(x))  # [1, 0, 0, 1, 0]
```

---

## Leaky ReLU

### Formula

```
Leaky ReLU(x) = x if x > 0
              = alpha * x if x <= 0  (alpha typically 0.01)
```

### Characteristics

- **Range**: (-∞, ∞)
- **Allows Small Negative**: Slope alpha for negative values
- **Solves Dying ReLU**: Maintains gradient for x < 0
- **Parameter**: alpha (usually 0.01)

### Graph (alpha=0.01)

```
       2 |
       0 |--/
      -2 |/
     -5  -2.5 0  2.5  5
```

### Pros
- Fixes dying ReLU problem
- Maintains gradients
- Still computationally efficient

### Cons
- Slightly more complex
- Alpha is hyperparameter to tune

### Usage
- When dying ReLU suspected
- Deep networks
- Can use with tuning

### Code Example

```python
import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

# Example
x = np.array([1, -2, 0, 3, -1])
print(leaky_relu(x))  # [1, -0.02, 0, 3, -0.01]
```

---

## ELU (Exponential Linear Unit)

### Formula

```
ELU(x) = x if x > 0
       = alpha * (e^x - 1) if x <= 0
```

### Characteristics

- **Range**: [-alpha, ∞)
- **Smooth**: Differentiable everywhere
- **Zero-Centered**: For negative x, approaches -alpha
- **Parameter**: alpha (usually 1)

### Pros
- Smoother than ReLU
- Zero-centered for negative inputs
- Better convergence

### Cons
- Computationally more expensive
- Less popular than ReLU
- Slower training

### Usage
- When smoothness matters
- Research/specific applications
- Less common in production

---

## Softmax Function

### Formula

```
Softmax(xi) = e^xi / Σ(e^xj) for all j
```

### Characteristics

- **Range**: (0, 1) per element
- **Sum**: All outputs sum to 1
- **Probability-like**: Valid probability distribution
- **Multi-class**: One output per class

### Example

```
Input: [2.0, 1.0, 0.1]

Exponents: [e^2, e^1, e^0.1] = [7.39, 2.72, 1.11]
Sum: 7.39 + 2.72 + 1.11 = 11.22

Output: [7.39/11.22, 2.72/11.22, 1.11/11.22]
       = [0.659, 0.243, 0.099]  (sums to 1)
```

### Pros
- True probability distribution
- Clear interpretation
- Ideal for multi-class

### Cons
- Only for output layer
- Computationally expensive
- Not for hidden layers

### Usage
- **Multi-class Classification**: Output layer only
- **Probability Output**: When probabilities needed
- **Mutually Exclusive**: Classes are exclusive

### Code Example

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Example
x = np.array([[2.0, 1.0, 0.1]])
print(softmax(x))  # [[0.659, 0.243, 0.099]]
```

---

## Comparison & Selection

### Function Comparison Table

| Function | Range | Vanishing Gradient | Zero-Centered | Speed | Use Case |
|----------|-------|-------------------|---------------|-------|----------|
| Sigmoid | (0,1) | Yes | No | Medium | Output (binary) |
| Tanh | (-1,1) | Yes | Yes | Medium | RNN/LSTM |
| ReLU | [0,∞) | No | No | Fast | Hidden (default) |
| Leaky ReLU | (-∞,∞) | No | Slight | Fast | Hidden (deep nets) |
| ELU | [-α,∞) | No | Yes | Slow | Hidden (smooth) |
| Softmax | (0,1) | No | - | Slow | Output (multi-class) |

### Selection Guide

**For Hidden Layers**:
1. Start with **ReLU**
2. If dying ReLU: Use **Leaky ReLU**
3. For deep networks: Try **ELU**
4. For RNNs: Use **tanh**

**For Output Layer**:
1. **Regression**: Linear (no activation)
2. **Binary Classification**: Sigmoid
3. **Multi-class**: Softmax
4. **Multi-label**: Sigmoid (for each output)

---

## Resources

### Key Papers
- "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio)
- "Delving Deep into Rectifiers" (He et al.)
- "Fast and Accurate Deep Network Learning" (ReLU paper)

### Further Reading
- TensorFlow Activation Functions Documentation
- PyTorch Activation Modules
- Deep Learning book - Chapter on Activation Functions

---

**Last Updated**: January 2026
**Difficulty Level**: Intermediate
**Prerequisites**: Basic calculus, neural network basics
