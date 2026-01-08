# Convolutional Neural Networks (CNN) Fundamentals

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Inspiration](#biological-inspiration)
3. [Core Concepts](#core-concepts)
4. [Architecture Components](#architecture-components)
5. [Mathematical Foundation](#mathematical-foundation)
6. [CNN vs Fully Connected Networks](#cnn-vs-fully-connected-networks)

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed to process grid-like data, particularly images. They leverage the spatial structure of images through local connectivity and weight sharing, making them highly effective for computer vision tasks.

**Key Applications:**
- Image classification (ImageNet, CIFAR-10, MNIST)
- Object detection (YOLO, R-CNN)
- Semantic segmentation
- Face recognition
- Medical image analysis
- Video analysis

## Biological Inspiration

CNNs are inspired by the visual cortex of the brain:

- **Simple Cells**: Detect local features (edges, textures) - similar to convolutional filters
- **Complex Cells**: Aggregate information over small regions - similar to pooling
- **Hierarchical Processing**: Lower layers detect simple patterns, higher layers detect complex patterns

Hubel and Wiesel (1968) discovered that neurons in the visual cortex are organized in a hierarchical manner with receptive fields.

## Core Concepts

### 1. Convolution Operation

```
Input Feature Map (28x28x3)  |  Filter (5x5x3)  |  Output (24x24x32)
  conv2D with 32 filters
```

Convolution applies a learnable filter across the input:
- **Filter Size**: Typically 3x3, 5x5, or 7x7
- **Stride**: Step size when sliding the filter (1 or 2 common)
- **Padding**: Adding zeros around input (same or valid)
- **Output Size**: (H - K + 2P) / S + 1
  - H: Input height
  - K: Kernel size
  - P: Padding
  - S: Stride

### 2. Pooling

Downsamples feature maps to reduce computation and capture dominant features:

**Max Pooling**:
```
Input (4x4)    Max Pool (2x2)    Output (2x2)
[1, 2]          [stride=2]        [2]
[3, 4]    →                   →   [4]
[5, 6]
[7, 8]
```

**Average Pooling**: Averages values instead of taking max

### 3. Activation Functions

**ReLU (Rectified Linear Unit)**:
- σ(x) = max(0, x)
- Most popular for CNNs
- Avoids vanishing gradient problem

**Softmax (Output Layer)**:
- For multi-class classification
- σ(x_i) = e^(x_i) / Σ e^(x_j)

## Architecture Components

### Input Layer
- Expects 3D or 4D tensors
- Dimensions: (Height, Width, Channels) for single image
- Dimensions: (Batch, Height, Width, Channels) for batch

### Convolutional Layer
```
Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')
```
- **Filters**: Number of feature maps to learn
- **Kernel Size**: Size of the filter
- **Activation**: Applied element-wise
- **Padding**: 'same' preserves spatial dimensions, 'valid' reduces them

### Pooling Layer
```
MaxPooling2D(pool_size=(2,2), strides=(2,2))
```
- Reduces spatial dimensions
- No learnable parameters
- Typical sizes: 2x2 or 3x3

### Batch Normalization
```
BatchNormalization()
```
- Normalizes layer inputs
- Speeds up training
- Allows higher learning rates
- Reduces internal covariate shift

### Dropout Layer
```
Dropout(rate=0.5)
```
- Randomly deactivates neurons during training
- Prevents overfitting
- Typical rates: 0.2-0.5

### Fully Connected Layer
```
Dense(units=128, activation='relu')
```
- Traditional neural network layer
- Used at end for classification
- Learns non-linear combinations

## Mathematical Foundation

### Convolution Formula

(I * K)[i, j] = ΣΣ I[i+m, j+n] × K[m, n]

Where:
- I: Input image
- K: Kernel/filter
- (i, j): Output position

### Parameter Count

For a Conv2D layer:
- Parameters = (K_h × K_w × C_in + 1) × C_out
  - K_h, K_w: Kernel height, width
  - C_in: Input channels
  - C_out: Output channels
  - +1: Bias term

Example: Conv2D(3×3, 32→64 filters, RGB input)
- Parameters = (3 × 3 × 32 + 1) × 64 = 18,496

## CNN vs Fully Connected Networks

### Advantages of CNN
✓ Local connectivity reduces parameters
✓ Weight sharing learns same features everywhere
✓ Leverages spatial structure of images
✓ Hierarchical feature learning
✓ Translation invariance (with pooling)
✓ Faster training and inference

### Example Comparison

**Fully Connected**:
- Input: 28×28 = 784 neurons
- Hidden layer 1: 512 neurons
- Parameters: 784 × 512 + 512 = 401,920

**CNN**:
- Input: 28×28×1
- Conv2D: 32 filters (3×3) = 320 parameters
- MaxPooling: 0 parameters
- Much more efficient!

## Classic CNN Architectures

### LeNet-5 (1998)
- First successful CNN
- 7 layers
- Used for MNIST digit recognition
- ~60,000 parameters

### AlexNet (2012)
- 8 layers (5 conv, 3 fully connected)
- Won ImageNet competition
- 60 million parameters
- Introduced ReLU activation

### VGG (2014)
- 16-19 layers
- Simple, uniform 3×3 convolutions
- 138 million parameters
- Effective feature extractor

### ResNet (2015)
- 50-152 layers with residual connections
- Skip connections prevent vanishing gradients
- Won ImageNet 2015

### Inception (GoogLeNet)
- Multi-scale feature extraction
- Inception modules with 1×1 convolutions
- More efficient parameter usage

## Training Tips

1. **Data Augmentation**: Rotate, flip, zoom, crop images
2. **Batch Normalization**: Use after conv layers
3. **Dropout**: Use before dense layers (0.5 typical)
4. **Learning Rate Scheduling**: Reduce over time
5. **Early Stopping**: Monitor validation loss
6. **Weight Initialization**: He initialization for ReLU

## Next Steps

- Understand convolution mechanics in detail
- Implement CNN from scratch
- Learn about pooling strategies
- Study famous architectures
- Apply CNNs to real datasets
