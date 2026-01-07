# Hyperparameter Tuning Exercise

## Objective
Learn how to systematically tune hyperparameters of a neural network to improve model performance.

## Key Hyperparameters to Tune

1. **Learning Rate**: Controls the step size during gradient descent
   - Typical range: [0.0001, 0.1]
   - Too high: Model diverges
   - Too low: Slow convergence

2. **Batch Size**: Number of samples per gradient update
   - Typical range: [16, 256]
   - Larger: Faster training, less noise
   - Smaller: Better generalization

3. **Number of Hidden Layers**: Network depth
   - 1-2 hidden layers for most tasks
   - More layers: Better feature extraction, more overfitting risk

4. **Hidden Layer Size**: Number of neurons per layer
   - Typical range: [32, 512]
   - More neurons: Better representation power

5. **Dropout Rate**: Fraction of neurons to drop
   - Typical range: [0.2, 0.5]
   - Prevents overfitting

6. **L2 Regularization (Weight Decay)**: Penalizes large weights
   - Typical range: [0.0001, 0.01]
   - Prevents overfitting

7. **Epochs**: Number of training iterations
   - Typically: 50-200
   - Use early stopping to find optimal value

## Tuning Strategies

### 1. Grid Search
Test all combinations of hyperparameter values
```python
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        # Train model and evaluate
```

### 2. Random Search
Randomly sample hyperparameter combinations
```python
import random
for _ in range(100):
    lr = random.uniform(0.0001, 0.1)
    bs = random.choice([16, 32, 64, 128])
    # Train model and evaluate
```

### 3. Bayesian Optimization
Use probabilistic model to guide search (advanced)

## Exercise Tasks

1. **Compare Learning Rates**
   - Train networks with LR = [0.0001, 0.001, 0.01, 0.1]
   - Plot train/validation curves
   - Find optimal LR

2. **Compare Network Sizes**
   - Test architectures with different hidden layer sizes
   - Compare training time and accuracy

3. **Effect of Dropout**
   - Train with dropout rates = [0.0, 0.3, 0.5, 0.7]
   - Observe overfitting reduction

4. **Grid Search (Optional)**
   - Perform grid search over 2-3 key hyperparameters
   - Document results in a table

## Expected Outcomes

- Understand how different hyperparameters affect model performance
- Develop intuition for good hyperparameter ranges
- Learn to balance model complexity vs generalization
