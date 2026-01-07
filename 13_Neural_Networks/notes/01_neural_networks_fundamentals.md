# Neural Networks Fundamentals

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Inspiration](#biological-inspiration)
3. [Basic Concepts](#basic-concepts)
4. [Perceptron Model](#perceptron-model)
5. [Multilayer Perceptron](#multilayer-perceptron)
6. [Network Architecture](#network-architecture)
7. [Forward Propagation](#forward-propagation)
8. [Key Components](#key-components)
9. [Applications](#applications)
10. [Resources](#resources)

---

## Introduction

Neural Networks are computational models inspired by biological neural networks in animal brains. They form the foundation of deep learning and have revolutionized machine learning, achieving state-of-the-art results in various domains.

### What are Neural Networks?

Neural networks are mathematical models consisting of interconnected nodes (neurons) organized in layers. They learn to make predictions by adjusting weights through a process called training.

### Why Neural Networks?

- **Non-linearity**: Can model complex, non-linear relationships
- **Feature Learning**: Automatically learn relevant features
- **Scalability**: Handle high-dimensional data
- **Universal Approximation**: Can approximate any continuous function
- **Parallel Processing**: Suit for GPU acceleration

---

## Biological Inspiration

### The Biological Neuron

A biological neuron consists of:
- **Soma (Cell Body)**: Contains the nucleus
- **Dendrites**: Receive signals from other neurons
- **Axon**: Sends signals to other neurons
- **Synapse**: Connection between neurons

### Artificial vs Biological

| Aspect | Biological | Artificial |
|--------|-----------|------------|
| Neurons | ~86 billion | Thousands to billions |
| Speed | Milliseconds | Nanoseconds |
| Learning | Hebbian | Gradient Descent |

---

## Basic Concepts

### Neuron (Node)

An artificial neuron performs computation:

```
Output = f(w1*x1 + w2*x2 + ... + wn*xn + b)
```

Where f is an activation function.

### Mathematical Representation

```
z = W*X + b   (weighted sum)
a = f(z)      (activation output)
```

### Example

```
Inputs: [2, 3]
Weights: [0.5, 0.7]
Bias: 0.2

z = (0.5 * 2) + (0.7 * 3) + 0.2 = 3.3
Output = sigmoid(3.3) ≈ 0.965
```

---

## Perceptron Model

### Definition

The Perceptron is the simplest neural network - a single neuron with binary output.

### Perceptron Learning Rule

```
If wrong prediction:
  w_new = w_old + learning_rate * (target - prediction) * input
```

### Limitations

- Only solves linearly separable problems
- Cannot learn XOR function
- Limited to single layer

---

## Multilayer Perceptron (MLP)

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
```

### Why Multiple Layers?

1. **Layer 1**: Learns simple features
2. **Layer 2**: Combines into higher-level features
3. **Layer 3+**: Learns abstract representations

### Universal Approximation Theorem

An MLP with:
- One hidden layer
- Sufficient hidden units  
- Non-linear activation

Can approximate ANY continuous function.

---

## Network Architecture

### Components

1. **Input Layer**: Number of neurons = number of features
2. **Hidden Layers**: Perform computation and learning
3. **Output Layer**: 
   - Regression: 1 neuron
   - Binary Classification: 1 neuron
   - Multiclass: n neurons (n = number of classes)

### Typical Ranges

- Number of hidden layers: 1-5+
- Neurons per layer: 50-1000
- Input features: 1-10,000+

---

## Forward Propagation

### Algorithm

```
For each layer l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activation(z[l])
Return a[L]
```

### Example Calculation

```
Input: [2, 1]
Layer 1: z1 = W1 @ [2,1] + b1
Layer 1 output: a1 = ReLU(z1)
Layer 2: z2 = W2 @ a1 + b2
Final: a2 = Sigmoid(z2)
```

---

## Key Components

### 1. Weights (W)
- Learnable parameters
- Initialize with small random values
- Updated during training

### 2. Biases (b)
- Learnable parameters
- Usually initialized to zero
- Allow function shift

### 3. Activation Functions

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
Range: (0, 1)
Use: Output layer (binary classification)
```

#### ReLU
```
f(x) = max(0, x)
Range: [0, ∞)
Use: Hidden layers (most popular)
```

#### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
Range: (-1, 1)
Use: Hidden layers
```

#### Softmax
```
f(xi) = e^xi / Σ(e^xj)
Range: (0, 1), sums to 1
Use: Multiclass classification
```

### 4. Loss Functions

#### Mean Squared Error
```
L = (1/m) * Σ(y_pred - y_true)²
Use: Regression
```

#### Cross-Entropy
```
L = -Σ(y_true * log(y_pred))
Use: Classification
```

---

## Applications

### Computer Vision
- Image Classification
- Object Detection
- Face Recognition
- Semantic Segmentation

### Natural Language Processing
- Text Classification
- Machine Translation
- Sentiment Analysis
- Named Entity Recognition

### Time Series
- Stock Price Prediction
- Weather Forecasting
- Anomaly Detection

### Other Applications
- Drug Discovery
- Game Playing
- Recommendation Systems
- Speech Recognition

---

## Resources

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Neural Networks and Deep Learning" by Nielsen
- "The Deep Learning Book" (free online)

### Online Courses
- Coursera: Deep Learning Specialization (Andrew Ng)
- Fast.ai: Practical Deep Learning
- Udacity: Deep Learning Nanodegree
- 3Blue1Brown: Neural Networks on YouTube

### Frameworks
- TensorFlow
- PyTorch
- Keras
- JAX

### Interactive Tools
- TensorFlow Playground
- ConvNetJS
- Neural Network Visualization

### Research Papers
- "A Brief History of Neural Networks" - Schmidhuber
- "Backpropagation and Architectures" - Rumelhart et al. (1986)
- "ImageNet Classification" - Krizhevsky et al. (2012)

### Datasets
- MNIST: Handwritten digits
- CIFAR-10: Small images
- ImageNet: Large-scale images
- Fashion-MNIST: Clothing items

---

## Summary

- Neurons are basic computational units
- Weights and biases are learnable parameters
- Activation functions introduce non-linearity
- Multiple layers enable hierarchical learning
- Forward propagation computes output from input
- MLPs can approximate any continuous function
- Deep learning uses many layers

---

**Next Topics**:
1. Activation Functions
2. Forward & Backpropagation
3. Training & Optimization
4. Regularization Techniques
5. Advanced Architectures

---

**Last Updated**: January 2026
**Difficulty Level**: Beginner to Intermediate
**Prerequisites**: Linear Algebra, Calculus, Python Basics
