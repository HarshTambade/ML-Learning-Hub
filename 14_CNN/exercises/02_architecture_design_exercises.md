# CNN Architecture Design Exercises

## Intermediate Level Exercises

### Exercise 1: Design a Custom CNN for CIFAR-100
**Objective:** Build a CNN architecture optimized for CIFAR-100 dataset (100 classes)

**Requirements:**
- Design at least 4 convolutional layers
- Use batch normalization after each conv layer
- Implement skip connections in at least 2 blocks
- Achieve >60% accuracy on test set

**Hints:**
- CIFAR-100 images are 32x32, start with 32 filters and increase
- Use residual blocks for better gradient flow
- Consider using adaptive average pooling before final layer

**Expected Output:**
```python
class CIFAR100CNN(nn.Module):
    def __init__(self):
        # Your architecture here
    
    def forward(self, x):
        # Forward pass
```

---

### Exercise 2: VGG-style Architecture Implementation
**Objective:** Implement a simplified VGG-style network

**Requirements:**
- Create repeating blocks of Conv-ReLU layers
- Use Max pooling after each block
- Design 3 blocks with increasing filters (64→128→256)
- Document layer dimensions at each stage

**Hints:**
- VGG uses 3x3 convolutions consistently
- Multiple convolutions before pooling
- Track feature map dimensions for debugging

**Bonus:** Implement architecture in a configurable way using a config dictionary

---

### Exercise 3: Inception Module Design
**Objective:** Create and use Inception modules in a CNN

**Requirements:**
- Implement InceptionBlock with 4 parallel paths
- Use 1x1 convolutions for dimensionality reduction
- Build a network with 2-3 Inception blocks
- Test on CIFAR-10

**Structure:**
```
Path 1: 1x1 conv
Path 2: 1x1 conv → 3x3 conv
Path 3: 1x1 conv → 5x5 conv
Path 4: MaxPool → 1x1 conv
```

---

### Exercise 4: Bottleneck Residual Blocks
**Objective:** Implement efficient bottleneck blocks

**Requirements:**
- Create a BottleneckBlock class
- Structure: 1x1 (reduce) → 3x3 → 1x1 (expand)
- Implement multiple bottleneck blocks in sequence
- Compare parameters with standard residual blocks

**Learning Goal:** Understand why bottleneck design reduces parameters

---

### Exercise 5: Mobile-Efficient CNN
**Objective:** Design a CNN with minimal parameters

**Requirements:**
- Use depthwise separable convolutions
- Target: <1M parameters
- Maintain >85% accuracy on CIFAR-10
- Profile model size and inference time

**Key Techniques:**
- Depthwise convolution + Pointwise convolution
- Bottleneck blocks
- Global average pooling

---

## Advanced Level Exercises

### Exercise 6: Neural Architecture Search (NAS)
**Objective:** Implement basic NAS for cell design

**Requirements:**
- Define a search space of operations
- Create a small CNN that can be parameterized
- Implement random search over architecture space
- Find best performing configuration

**Operations to Consider:**
- Conv 1x1, 3x3, 5x5
- Separable Conv
- Max/Avg Pool
- Skip connection

---

### Exercise 7: Multi-Scale Feature Extraction
**Objective:** Build CNN with feature pyramids

**Requirements:**
- Extract features at multiple scales
- Implement Feature Pyramid Network (FPN) structure
- Combine features from different levels
- Test on object detection task

**Architecture Components:**
- Bottom-up pathway (backbone)
- Top-down pathway (upsampling)
- Lateral connections

---

### Exercise 8: Efficient Channel Design
**Objective:** Optimize channel dimensions per layer

**Requirements:**
- Start with baseline model
- Systematically reduce channels in each layer
- Maintain accuracy while reducing parameters
- Create a channel scaling function

**Analysis:**
- Plot accuracy vs model size
- Identify bottleneck layers
- Determine optimal channel configuration

---

## Design Patterns to Implement

### Pattern 1: Residual Connections
```python
class ResidualBlock(nn.Module):
    # Implement skip connection architecture
```

### Pattern 2: Dense Connections
```python
class DenseBlock(nn.Module):
    # Each layer connects to all previous layers
```

### Pattern 3: Squeeze-Excitation
```python
class SEBlock(nn.Module):
    # Channel attention mechanism
```

---

## Architecture Analysis Tasks

### Task 1: Parameter Count Analysis
- Design 3 different architectures
- Calculate total parameters for each
- Measure inference time
- Compare accuracy vs efficiency trade-offs

### Task 2: Feature Map Visualization
- Build a CNN
- Extract intermediate feature maps
- Visualize learned filters
- Analyze feature complexity across layers

### Task 3: Gradient Flow Analysis
- Compare gradient flow with/without skip connections
- Measure gradient magnitude at each layer
- Demonstrate why residual networks work better

---

## Implementation Checklist

- [ ] Design architecture with clear layer definitions
- [ ] Implement forward pass correctly
- [ ] Calculate and document parameter count
- [ ] Test on sample data (shape verification)
- [ ] Train and evaluate on dataset
- [ ] Profile computational efficiency
- [ ] Document architectural choices
- [ ] Compare with baseline models

---

## Evaluation Criteria

**Code Quality (20%)**
- Clean, readable code
- Proper documentation
- Efficient implementation

**Architecture Design (30%)**
- Novel or well-justified design choices
- Proper use of CNN components
- Scalability and modularity

**Performance (30%)**
- Achieves target accuracy
- Efficient parameter usage
- Fast inference time

**Analysis (20%)**
- Understanding of design decisions
- Comparison with baselines
- Insights about trade-offs

---

## Resources for Architecture Design

- ResNet paper: "Deep Residual Learning for Image Recognition"
- Inception paper: "Going Deeper with Convolutions"
- MobileNet paper: "MobileNets: Efficient Convolutional Neural Networks"
- DenseNet paper: "Densely Connected Convolutional Networks"

---

## Expected Completion Time

- Basic exercises: 2-3 hours each
- Advanced exercises: 4-6 hours each
- Analysis tasks: 1-2 hours each
