# Practical CNN Training Tips and Best Practices

## Table of Contents
1. Data Preparation Best Practices
2. Training Workflow and Debugging
3. Hyperparameter Tuning Strategies
4. Common Pitfalls and Solutions
5. Performance Monitoring
6. Inference Optimization
7. Production Deployment Checklist
8. Case Studies and Real-world Examples

## 1. Data Preparation Best Practices

### Image Normalization
- Use ImageNet statistics for pre-trained models: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- For custom datasets, compute statistics on training data
- Always apply same normalization to test data

### Data Splitting
- **Training (60-70%)**: Learn model parameters
- **Validation (10-15%)**: Tune hyperparameters
- **Test (15-20%)**: Final evaluation
- Use stratified sampling for imbalanced datasets

### Data Augmentation Strategy
```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
```

## 2. Training Workflow and Debugging

### Training Loop Template
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)
```

### Debug Checklist
- [ ] Verify input shape: (batch_size, channels, height, width)
- [ ] Check loss is decreasing
- [ ] Verify gradients are not NaN or Inf
- [ ] Monitor learning rate decay
- [ ] Track training vs validation loss divergence
- [ ] Use gradient clipping if needed

## 3. Hyperparameter Tuning Strategies

### Learning Rate Selection
- **Start with 0.001** for Adam optimizer
- **Use learning rate finder**: increase LR exponentially, plot loss
- **For SGD**: typically 0.01-0.1
- **Schedule decay**: reduce LR by 10x every 30% of epochs

### Batch Size Impact
- **Larger batches** (256-512): Better gradient estimates, less frequent updates
- **Smaller batches** (32-64): More frequent updates, better generalization
- **Typical practice**: 64 or 128 for ImageNet-scale datasets

### Optimizer Selection
- **Adam**: Best default choice for CNNs, adaptive learning rates
- **SGD+Momentum**: Better generalization with proper LR scheduling
- **AdamW**: Adam with weight decay, prevents overfitting

### Model Architecture Choices
- **For small datasets**: Use transfer learning with pre-trained models
- **For large datasets**: Train from scratch or fine-tune
- **Depth vs Width**: Deeper networks better for complex patterns

## 4. Common Pitfalls and Solutions

### Overfitting
**Symptoms**: Training accuracy high, validation accuracy low
**Solutions**:
- Increase data augmentation
- Add dropout layers
- Reduce model capacity
- Use L2 regularization
- Increase training data

### Underfitting
**Symptoms**: Both training and validation accuracy low
**Solutions**:
- Increase model capacity
- Reduce regularization
- Train for more epochs
- Use more complex architecture
- Improve data quality

### Gradient Issues
**Exploding gradients**: Loss becomes NaN
**Solution**: Gradient clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Vanishing gradients**: Model stops learning
**Solution**: Use batch normalization, skip connections

## 5. Performance Monitoring

### Key Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

### Visualization Tools
```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Plot training curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap='Blues')
```

## 6. Inference Optimization

### Model Quantization
- Convert FP32 to INT8: 4x smaller, faster inference
- Minimal accuracy loss with proper calibration

### ONNX Export
```python
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['images'],
                  output_names=['predictions'])
```

### Batch Inference
- Process multiple images together
- Utilize GPU parallelism
- Typical batch size: 32-256

## 7. Production Deployment Checklist

- [ ] Model inference time < 100ms per image
- [ ] Model size < 500MB (or quantize to <100MB)
- [ ] Accuracy on test set > baseline
- [ ] Input/output specifications documented
- [ ] Error handling and edge cases tested
- [ ] Model versioning system in place
- [ ] Monitoring and logging configured
- [ ] Gradual rollout plan prepared

## 8. Quick Reference: Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training
for epoch in range(100):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Val Loss: {val_loss/len(val_loader):.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
```

## Key Takeaways
- **Data quality matters more than quantity**: Proper preprocessing and augmentation
- **Monitor everything**: Loss, accuracy, gradients, learning rate
- **Transfer learning is powerful**: Pre-trained models save time and improve results
- **Reproducibility is essential**: Fix random seeds, log hyperparameters
- **Start simple, iterate**: Baseline model first, then optimize
