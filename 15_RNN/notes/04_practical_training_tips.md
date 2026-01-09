# Practical Training Tips for RNNs

## Introduction

Training RNNs effectively requires understanding of practical considerations beyond theory. This guide covers troubleshooting, hyperparameter tuning, and best practices based on real-world experience.

## Data Preparation

### Sequence Length Handling

1. **Padding Short Sequences**
   - Pad sequences to fixed length with special token (0 or -1)
   - Use masking to ignore padded positions in loss calculation
   - Alternative: Dynamic padding by creating mini-batches with similar lengths

2. **Handling Long Sequences**
   - Truncate very long sequences to manageable length (512-1024 tokens)
   - Use sliding window approach for document-level tasks
   - Consider hierarchical RNNs for very long documents

3. **Bucketing Strategy**
   - Group sequences by similar lengths in batches
   - Reduces padding overhead and improves efficiency
   - Particularly effective for variable-length sequences

### Normalization and Scaling

1. **Input Normalization**
   - Normalize numerical features to mean=0, std=1
   - Use StandardScaler or z-score normalization
   - Critical for stable gradient flow

2. **Embedding Scaling**
   - Initialize embeddings with appropriate variance
   - Use Xavier/Glorot initialization: variance = 1/input_dim
   - Prevents vanishing/exploding gradients in early training

## Hyperparameter Tuning

### Learning Rate

- **Initial value**: Start with 0.001-0.01 and adjust based on loss curves
- **Learning rate decay**: Reduce LR by 0.5 every 5-10 epochs if loss plateaus
- **Adaptive optimizers**: Adam often works better than SGD for RNNs
- **Warmup**: Gradually increase LR from 0 for first few epochs

### Batch Size

- **Larger batches**: More stable gradients but higher memory
- **Recommended**: 32-128 for most tasks
- **Small datasets**: Use smaller batches (16-32) to reduce overfitting
- **Trade-off**: Larger batch with fewer epochs vs smaller batch with more epochs

### Hidden Size

- **Rule of thumb**: 128-512 hidden units for most tasks
- **Larger models**: Use for larger datasets and complex patterns
- **Smaller models**: Better for small datasets and real-time inference
- **Progressive**: Start small, increase if underfitting persists

### Number of Layers

- **Shallow**: 1-2 layers sufficient for simpler tasks
- **Deep**: 2-4 layers for complex patterns
- **Stack with caution**: Adding layers beyond 4 rarely helps, risks vanishing gradients
- **Residual connections**: Essential for very deep RNNs (>3 layers)

### Dropout Rate

- **Typical range**: 0.3-0.5
- **No dropout**: If underfitting
- **High dropout (0.7+)**: Risk of underfitting
- **Recurrent dropout**: Separate parameter, usually 0.1-0.3

## Training Strategies

### Gradient Clipping

```python
# Essential for RNN training
if ||gradients|| > threshold:
    gradients = gradients * (threshold / ||gradients||)
```

- **Threshold**: Usually 1.0 or 5.0
- **When**: Apply always for safety
- **Effect**: Prevents exploding gradients without affecting learning

### Early Stopping

- Monitor validation loss every epoch
- Stop if no improvement for 10-20 epochs
- Prevents overfitting and saves training time
- Save best model weights when validation loss improves

### Warm-up and Cool-down

- **Warm-up**: Gradually increase learning rate for 1-5 epochs
- **Cool-down**: Reduce learning rate in final epochs
- Improves final model stability and performance

## Troubleshooting Common Issues

### Loss is NaN or Inf

**Causes**:
- Learning rate too high
- Gradients exploding (especially with long sequences)
- Numerical issues in loss computation

**Solutions**:
- Reduce learning rate (by 10x)
- Enable gradient clipping if not already
- Check data for extreme values
- Verify loss function implementation

### Training loss decreases but validation loss increases

**Problem**: Overfitting

**Solutions**:
- Add dropout (increase from 0.3 to 0.5)
- Add regularization (L2 penalty)
- Reduce model size
- Increase training data or augmentation
- Use early stopping

### Training loss plateaus early

**Problem**: Underfitting

**Solutions**:
- Increase learning rate
- Increase model capacity (hidden size, layers)
- Reduce dropout
- Train for more epochs
- Check if data is preprocessed correctly

### Very slow training

**Causes**:
- Sequences too long
- Model too large
- Batch size too small
- GPU memory swapping

**Solutions**:
- Truncate sequences to reasonable length (512-1024)
- Use smaller model or fewer layers
- Increase batch size if memory allows
- Profile code to find bottleneck

## Implementation Best Practices

### Model Architecture

1. **Use PyTorch nn.LSTM or nn.GRU**: Highly optimized, bidirectional support
2. **Pack padded sequences**: Improves efficiency and correctness
3. **Add layer normalization**: Stabilizes training
4. **Use residual connections**: For deep networks (>2 layers)

### Code Structure

```python
# Good structure:
model = nn.Sequential(
    nn.Embedding(vocab_size, embed_dim),
    nn.LSTM(embed_dim, hidden_size, num_layers=2, 
            batch_first=True, dropout=0.3, bidirectional=True),
    nn.Linear(hidden_size*2, output_size)
)
```

### Training Loop

1. Create train/validation/test splits
2. Normalize data
3. Create data loaders with appropriate batch size
4. Implement gradient clipping
5. Use Adam optimizer with weight decay
6. Monitor multiple metrics (loss, accuracy, etc.)
7. Save checkpoints regularly

## Common Patterns by Task

### Classification (Single Output)
- Use many-to-one: feed final hidden state to classifier
- Use attention: weighted sum of all hidden states

### Sequence Labeling (Token-level)
- Use many-to-many: predict label for each token
- Use bidirectional RNN for context from both directions

### Sequence Generation
- Use encoder-decoder with attention
- Use teacher forcing during training
- Use beam search or sampling during inference

### Time Series Forecasting
- Use many-to-one or many-to-many depending on task
- Normalize targets carefully
- Use appropriate loss function (MSE for regression)

## Performance Optimization

### Memory Efficiency
- Use smaller hidden sizes if memory limited
- Reduce sequence length if possible
- Use mixed precision training (float16)
- Gradient accumulation for large batch sizes

### Speed
- Profile code to find bottlenecks
- Use fused LSTM/GRU implementations
- Batch processing when possible
- GPU acceleration critical for RNNs

## Debugging Checklist

- [ ] Data loading correctly?
- [ ] Shapes match expectations?
- [ ] Gradients flowing backwards?
- [ ] Loss reasonable magnitude?
- [ ] Validation loss improving?
- [ ] Training time reasonable?
- [ ] Predictions sensible?
- [ ] Hyperparameters tuned?

## Learning Objectives

After studying this section, you should be able to:
- Prepare data appropriately for RNNs
- Tune hyperparameters effectively
- Recognize and fix common training issues
- Implement best practices for production models
- Debug RNN training problems systematically
