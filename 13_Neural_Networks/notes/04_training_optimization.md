# Training and Optimization

## Table of Contents
1. [Gradient Descent](#gradient-descent)
2. [Learning Rate](#learning-rate)
3. [Optimizers](#optimizers)
4. [Batch Processing](#batch-processing)
5. [Regularization](#regularization)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Training Monitoring](#training-monitoring)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)

---

## Gradient Descent

### Variants

**Batch Gradient Descent**
- Use entire dataset for each update
- Pros: Stable, accurate gradient
- Cons: Slow, memory intensive

**Stochastic Gradient Descent (SGD)**
- Use single sample for each update
- Pros: Fast, escapes local minima
- Cons: Noisy, oscillates

**Mini-batch Gradient Descent**
- Use small batch (32-256 samples)
- Pros: Balance between speed and stability
- Cons: Hyperparameter to tune

### Update Rule

```
W = W - learning_rate * gradient
W_new = W_old - lr * dL/dW
```

---

## Learning Rate

### Impact

**Too Small**
- Training very slow
- May get stuck in local minima
- Takes many epochs

**Too Large**
- Overshoots optimal weights
- Loss oscillates or diverges
- Training unstable

**Good Value**
- Steady convergence
- Reaches minimum efficiently
- Loss decreases each epoch

### Learning Rate Schedules

**Constant**: Fixed learning rate throughout

**Step Decay**: Decrease by factor every N epochs
```
lr = initial_lr * decay_rate^(epoch / decay_steps)
```

**Exponential Decay**: Exponential decrease
```
lr = initial_lr * e^(-decay_constant * epoch)
```

**Cosine Annealing**: Cosine schedule
```
lr = 0.5 * initial_lr * (1 + cos(epoch * pi / total_epochs))
```

---

## Optimizers

### SGD with Momentum

```
v = beta * v + (1 - beta) * gradient
W = W - learning_rate * v
```

**Benefits**:
- Accumulates gradient direction
- Faster convergence
- Better for noisy gradients

### RMSprop

```
v = beta * v + (1 - beta) * gradient^2
W = W - (learning_rate / sqrt(v + eps)) * gradient
```

**Benefits**:
- Adapts learning rate per parameter
- Handles varying gradient magnitudes

### Adam (Adaptive Moment Estimation)

```
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
W = W - learning_rate * m_hat / (sqrt(v_hat) + eps)
```

**Default parameters**: beta1=0.9, beta2=0.999

**Benefits**:
- Combines momentum and RMSprop
- Works well across problems
- **Most popular optimizer**

### Comparison

| Optimizer | Speed | Memory | Tuning |
|-----------|-------|--------|--------|
| SGD | Fast | Low | Hard |
| Momentum | Medium | Low | Medium |
| RMSprop | Medium | Medium | Medium |
| Adam | Medium | Medium | Easy |

---

## Batch Processing

### Batch Size Effects

**Small Batches (8-32)**
- Noisy gradients
- Better generalization
- More updates per epoch
- Needs smaller learning rate

**Large Batches (256-1024)**
- Smooth gradients
- May overfit
- Fewer updates per epoch
- Better parallelization

### Typical Ranges
- Small datasets: 32-64
- Medium datasets: 64-128
- Large datasets: 256-512
- GPU memory: 1024+

---

## Regularization

### L1 Regularization

```
L_total = L + lambda * sum(|W|)
```

- Encourages sparse weights
- Feature selection
- Some weights become zero

### L2 Regularization (Weight Decay)

```
L_total = L + lambda * sum(W^2)
```

- Prevents large weights
- Smoother functions
- All weights reduced

### Dropout

```
During training: Randomly drop neurons with probability p
During inference: Use all neurons, scale by (1-p)
```

- Prevents co-adaptation
- Ensemble effect
- Typical p: 0.2-0.5

### Early Stopping

```
Monitor validation loss
Stop when validation loss increases
Keep weights with best validation performance
```

---

## Hyperparameter Tuning

### Grid Search

```
Try all combinations of hyperparameters
Pros: Complete exploration
Cons: Exponential complexity
```

### Random Search

```
Randomly sample hyperparameters
Pros: Efficient, covers space
Cons: May miss good regions
```

### Bayesian Optimization

```
Use previous results to guide search
Sample promising regions
Pros: Sample efficient
Cons: Complex implementation
```

---

## Training Monitoring

### Key Metrics

**Training Loss**
- Should decrease consistently
- Indicates learning progress

**Validation Loss**
- Monitor overfitting
- Should decrease, then plateau
- Stop if starts increasing

**Accuracy/Metrics**
- Final performance measure
- Check on validation set

### Debugging Training

**Loss not decreasing**
- Learning rate too small
- Bad model initialization
- Data preprocessing issues

**Loss diverging**
- Learning rate too large
- Gradient explosion
- Check for NaN values

**Overfitting**
- Gap between train and validation loss
- Add regularization
- Use more data
- Reduce model complexity

---

## Best Practices

1. **Normalize inputs**: Zero mean, unit variance
2. **Initialize weights**: Xavier or He initialization
3. **Use batch normalization**: Stabilizes training
4. **Monitor metrics**: Use TensorBoard, Wandb
5. **Save checkpoints**: Keep best model
6. **Cross-validate**: Ensure generalization
7. **Start simple**: Add complexity gradually
8. **Tune learning rate**: Most important hyperparameter
9. **Use appropriate optimizer**: Adam is default
10. **Patience with training**: Deep learning needs time

---

## Code Examples

### PyTorch Training Loop

```python
import torch
import torch.nn as nn
from torch.optim import Adam

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # Forward
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_loss = 0
        for val_x, val_y in val_loader:
            val_pred = model(val_x)
            val_loss += loss_fn(val_pred, val_y)
        print(f'Epoch {epoch}: Loss={loss.item():.4f}, Val={val_loss:.4f}')
```

### TensorFlow Training

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=10,
    batch_size=32
)
```

---

**Last Updated**: January 2026
**Difficulty Level**: Intermediate
**Prerequisites**: Understanding of backpropagation and gradient descent
