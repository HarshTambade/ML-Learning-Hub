# CNN Architectures: From LeNet to ResNet

## LeNet-5 (1998)

**Yann LeCun et al. - Foundational CNN**

```
Input (32x32x1)
  ↓
Conv2D (6 filters, 5x5) → Sigmoid → AvgPool (2x2)
  ↓
Conv2D (16 filters, 5x5) → Sigmoid → AvgPool (2x2)
  ↓
Conv2D (120 filters, 5x5) → Sigmoid
  ↓
Dense (84) → Sigmoid
  ↓
Dense (10) → Softmax
```

- **Parameters**: ~60K
- **Use**: MNIST digit recognition
- **Achievement**: 99.2% accuracy

## AlexNet (2012)

**Krizhevsky et al. - ImageNet Competition Winner**

```
Input (227x227x3)
  ↓
Conv2D (96, 11x11, stride=4) + ReLU + MaxPool (3x3)
  ↓
Conv2D (256, 5x5) + ReLU + MaxPool (3x3)
  ↓
Conv2D (384, 3x3) + ReLU
  ↓
Conv2D (384, 3x3) + ReLU
  ↓
Conv2D (256, 3x3) + ReLU + MaxPool (3x3)
  ↓
Dense (4096) + ReLU + Dropout (0.5)
Dense (4096) + ReLU + Dropout (0.5)
Dense (1000) + Softmax
```

- **Parameters**: 60M
- **Key Innovation**: ReLU activation (faster training)
- **Dataset**: ImageNet (1.2M images, 1000 classes)
- **Top-1 Accuracy**: 62.5% (2012)

## VGG (2014)

**Very Deep Convolutional Networks - Simonyan & Zisserman**

```
All 3x3 Convolutions (uniform architecture)

VGG-16:
- Input (224x224x3)
- Block 1: 2x Conv(64) + MaxPool
- Block 2: 2x Conv(128) + MaxPool
- Block 3: 3x Conv(256) + MaxPool
- Block 4: 3x Conv(512) + MaxPool
- Block 5: 3x Conv(512) + MaxPool
- FC: Dense(4096) + ReLU + Dropout
- FC: Dense(4096) + ReLU + Dropout
- Output: Dense(1000) + Softmax
```

- **VGG-16**: 138M parameters
- **VGG-19**: 144M parameters
- **Key Insight**: Stack of small 3x3 filters equivalent to large filters
- **3x3 vs 5x5**: Two 3x3 filters = one 5x5 (fewer parameters, same receptive field)
- **Use**: Feature extractor for transfer learning

## ResNet (2015)

**Residual Networks - He et al.**

**Problem Solved**: Vanishing gradients in very deep networks

```
Basic Residual Block:
  x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU → y
  └────────────────────────────────┘
           (skip connection)
```

- **Identity Mapping**: y = F(x) + x (vs y = F(x))
- **ResNet-50**: 50 layers, 25.5M parameters
- **ResNet-101**: 101 layers, 44.5M parameters
- **ResNet-152**: 152 layers, 60M parameters
- **ImageNet Top-1**: 76.15% (ResNet-152)

## GoogLeNet / Inception (2014)

**Multi-scale Feature Extraction**

```
Inception Module:
            Input
      /    |    |    \
    1x1  3x3  5x5  MaxPool
    |     |     |     |
   Conv  Conv  Conv  Conv
     \    |    |    /
      Concatenate → Output
```

- **1x1 Convolutions**: Dimension reduction
- **Multiple Scales**: Capture features at different scales
- **Efficiency**: 1.4B FLOPS vs 15.3B for VGG
- **Parameters**: 6M (much fewer than VGG)

## MobileNet (2017)

**Efficient CNN for Mobile Devices**

```
Depthwise Separable Convolution:
  - Depthwise: 3x3 convolution per channel
  - Pointwise: 1x1 convolution to mix channels
  - 8-9x fewer parameters than standard conv
```

- **Size**: 16-17MB (vs 500MB for VGG)
- **Latency**: ~50ms on mobile
- **Accuracy**: 70.6% top-1 on ImageNet

## EfficientNet (2019)

**Compound Model Scaling**

```
Accuracy/Efficiency Trade-off:
  EfficientNet-B0: 77.1% accuracy, 390M FLOPs
  EfficientNet-B7: 84.4% accuracy, 37B FLOPs
```

**Scaling**: Balance depth, width, resolution
- Depth: More layers → better feature hierarchy
- Width: More channels → finer-grained features
- Resolution: Higher input → more detail

## Architecture Comparison

| Model | Year | Params | Top-1 | Speed |
|-------|------|--------|-------|-------|
| LeNet-5 | 1998 | 60K | 99.2% | ✓✓✓ |
| AlexNet | 2012 | 60M | 62.5% | ✓✓ |
| VGG-16 | 2014 | 138M | 71.3% | ✓ |
| GoogLeNet | 2014 | 6M | 69.8% | ✓✓✓ |
| ResNet-50 | 2015 | 25.5M | 76.1% | ✓✓ |
| MobileNet | 2017 | 4.3M | 70.6% | ✓✓✓ |
| EfficientNet-B0 | 2019 | 5.3M | 77.1% | ✓✓✓ |

## Design Principles

1. **Depth vs Width Trade-off**
2. **Skip Connections**: Enable very deep networks
3. **Multi-scale Processing**: Inception modules
4. **Efficiency**: MobileNet, EfficientNet
5. **Transfer Learning**: Pre-trained ImageNet models

## Which to Use?

- **Fast Prototyping**: ResNet-50
- **Small Model**: MobileNet
- **Best Accuracy**: EfficientNet-B7
- **Transfer Learning**: VGG (excellent features)
- **Production**: EfficientNet, MobileNet
