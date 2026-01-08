# CNN Code Examples

Comprehensive collection of PyTorch implementations demonstrating various CNN concepts and techniques.

## Files Overview

### 1. 01_basic_cnn_mnist.py
**Basic CNN Implementation on MNIST**
- Simple but complete CNN architecture
- Training loop with validation
- Loss tracking and visualization
- Perfect for beginners to understand CNN fundamentals

**Key Concepts:**
- Conv2d layers
- MaxPooling
- Fully connected layers
- Training and validation splits

### 2. 02_cnn_with_regularization.py
**CNN with Advanced Regularization Techniques**
- Dropout layers for preventing overfitting
- Batch normalization for stable training
- L2 regularization (weight_decay)
- Early stopping mechanism
- Learning rate scheduling
- Gradient clipping

**Key Concepts:**
- Regularization techniques
- Batch normalization
- Learning rate scheduling
- Early stopping
- Model evaluation metrics

### 3. 03_transfer_learning_example.py
**Transfer Learning with Pre-trained Models**
- Uses pre-trained ResNet50 model
- Two-phase training approach
- Frozen and fine-tuned backbone
- Custom classification head
- CIFAR10 dataset example

**Key Concepts:**
- Transfer learning paradigm
- Fine-tuning vs feature extraction
- Pre-trained model adaptation
- Two-stage training

### 4. 04_data_augmentation.py
**Comprehensive Data Augmentation Strategies**
- RandomHorizontalFlip
- RandomRotation
- ColorJitter for brightness/contrast/saturation
- RandomAffine for geometric transformations
- GaussianBlur
- RandomPerspective
- Augmentation visualization

**Key Concepts:**
- Data augmentation techniques
- Transforms pipeline
- Impact on model generalization
- Training vs test augmentation

### 5. 05_custom_cnn_architecture.py
**Custom CNN Architectures**
- ResidualBlock implementation (skip connections)
- InceptionBlock (parallel convolutions)
- CustomResNet with residual layers
- CustomInceptionNet
- DenseBlock with feature reuse

**Key Concepts:**
- Residual connections
- Inception modules
- Dense connections
- Architecture design patterns
- Feature reuse strategies

### 6. 06_batch_normalization_example.py
**Batch Normalization Impact Analysis**
- Comparison: CNN with and without BN
- Training dynamics comparison
- Loss curves visualization
- Accuracy improvement analysis
- Statistical normalization benefits

**Key Concepts:**
- Batch normalization mechanism
- Impact on training stability
- Convergence speed improvement
- Reduced internal covariate shift

## Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

## Quick Start

### Running Individual Examples

```bash
# Basic CNN on MNIST
python 01_basic_cnn_mnist.py

# Regularization techniques
python 02_cnn_with_regularization.py

# Transfer learning
python 03_transfer_learning_example.py

# Data augmentation
python 04_data_augmentation.py

# Custom architectures
python 05_custom_cnn_architecture.py

# Batch normalization comparison
python 06_batch_normalization_example.py
```

## Learning Path

1. **Beginners**: Start with `01_basic_cnn_mnist.py`
2. **Intermediate**: Explore `02_cnn_with_regularization.py` and `04_data_augmentation.py`
3. **Advanced**: Learn from `03_transfer_learning_example.py` and `05_custom_cnn_architecture.py`
4. **Optimization**: Study `06_batch_normalization_example.py`

## Key Concepts Covered

- **Convolution Operations**: Feature extraction through learned filters
- **Pooling**: Dimensionality reduction and feature selection
- **Regularization**: Dropout, L2 regularization, batch normalization
- **Optimization**: Adam optimizer, learning rate scheduling, early stopping
- **Data Augmentation**: Preventing overfitting through diverse training data
- **Transfer Learning**: Leveraging pre-trained models for new tasks
- **Architecture Design**: Skip connections, inception modules, dense connections
- **Training Dynamics**: Batch normalization effects on convergence

## Model Architecture Components

### Common Layers
- `Conv2d`: 2D convolution for feature extraction
- `BatchNorm2d`: Normalization across feature maps
- `MaxPool2d`: Max pooling for downsampling
- `Dropout`: Random neuron deactivation for regularization
- `Linear`: Fully connected layers for classification

### Activation Functions
- `ReLU`: Rectified Linear Unit (most common)
- `Softmax`: For multi-class probability distribution

## Performance Tips

1. **GPU Acceleration**: Use CUDA when available
2. **Batch Processing**: Experiment with different batch sizes
3. **Learning Rate**: Use learning rate scheduling for better convergence
4. **Early Stopping**: Prevent overfitting by monitoring validation metrics
5. **Data Augmentation**: Improve generalization without more data
6. **Batch Normalization**: Accelerate training and improve stability

## Hyperparameter Guidelines

- **Learning Rate**: 0.001 - 0.01 (adjust with scheduler)
- **Batch Size**: 32 - 128 depending on GPU memory
- **Dropout Rate**: 0.3 - 0.5 for regularization
- **Weight Decay**: 1e-4 - 1e-5 for L2 regularization
- **Epochs**: 10 - 50 depending on convergence

## Dataset Information

- **MNIST**: 60,000 training, 10,000 test, 28x28 grayscale, 10 classes
- **CIFAR10**: 50,000 training, 10,000 test, 32x32 RGB, 10 classes

## Expected Results

- MNIST: >98% accuracy after 10 epochs
- CIFAR10 (with transfer learning): >85% accuracy
- Batch norm typically improves accuracy by 2-5%

## Further Reading

- [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html)
- [CNN Architectures Explained](https://towardsdatascience.com/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Data Augmentation Best Practices](https://torchvision.readthedocs.io/)

## Contributing

Feel free to:
- Add new CNN architectures
- Implement additional regularization techniques
- Optimize performance
- Add comments and documentation
- Create variations for different datasets

## License

MIT License - Feel free to use for learning and projects
