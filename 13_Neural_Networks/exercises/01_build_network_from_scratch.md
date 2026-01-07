# Exercise: Build a Neural Network from Scratch

## Objective
Implement a 2-layer neural network for binary classification without using deep learning frameworks.

## Problem Statement
Given a dataset with 2 input features, build a neural network to classify points into 2 classes.

## Requirements

### Part 1: Data Preparation
- Generate or load a simple 2D dataset (moon dataset or circle dataset)
- Visualize the data
- Normalize/scale the features to [-1, 1]
- Split into train (80%) and test (20%)

### Part 2: Network Architecture
Implement a network with:
- Input layer: 2 neurons (features)
- Hidden layer: 4 neurons with sigmoid activation
- Output layer: 1 neuron with sigmoid activation

### Part 3: Forward Propagation
- Implement matrix operations for:
  - z = Wx + b
  - a = sigmoid(z)
- Handle batch inputs correctly

### Part 4: Backward Propagation
- Calculate gradients using chain rule
- Implement weight and bias updates

### Part 5: Training
- Train for 1000 iterations
- Use learning rate of 0.1
- Track loss every 100 iterations
- Plot loss curve

### Part 6: Evaluation
- Calculate accuracy on test set
- Visualize decision boundary
- Show predictions on test data

## Hints

```python
# Sigmoid function
sigreid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)

# Cross-entropy loss
loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
```

## Success Criteria
- Training loss decreases over iterations
- Test accuracy > 80%
- Decision boundary separates classes reasonably
- Code is well-commented

## Bonus
- Try different architectures (different layer sizes)
- Implement momentum for updates
- Try different activation functions (tanh, ReLU)
- Add batch normalization concepts
