# Transformer Architecture Fundamentals

## Overview

Transformers are a revolutionary deep learning architecture introduced by Vaswani et al. in "Attention Is All You Need" (2017). They completely replace recurrence with attention mechanisms and have become the foundation for state-of-the-art NLP and multimodal models.

## Key Innovation: Self-Attention

The core innovation of Transformers is the **self-attention mechanism**, which allows the model to:
- Focus on relevant parts of the input sequence
- Process all positions in parallel (unlike RNNs)
- Capture long-range dependencies efficiently
- Scale to very large datasets

## Core Components

### 1. Multi-Head Self-Attention

**Purpose**: Allow the model to attend to different representation subspaces at different positions.

**Mathematical Formulation**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

Multi-Head(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Parameters**:
- Q (Query), K (Key), V (Value): Projections of input
- d_k: Dimension of key
- h: Number of attention heads
- W^O: Output projection

**Advantages**:
- Each head learns different attention patterns
- Parallel computation across heads
- Captures multiple types of relationships simultaneously

### 2. Feed-Forward Networks (FFN)

**Architecture**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Characteristics**:
- Two linear layers with ReLU activation
- Applied independently to each position
- Adds non-linearity and expressive power
- Typically 4x larger hidden dimension

### 3. Layer Normalization

**Purpose**: Stabilize training and improve convergence

**Application**: Applied before or after attention/FFN
- Pre-norm (Applied before): Normalizes inputs
- Post-norm (Applied after): Normalizes outputs

### 4. Positional Encoding

**Problem**: Self-attention is permutation-invariant, doesn't encode position information

**Solution**: Add positional encodings to input embeddings

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Benefits**:
- Encodes absolute and relative positions
- Allows extrapolation to longer sequences
- Learnable alternative: Position embeddings

## Transformer Architecture

### Encoder Stack
- Multiple identical layers (typically 12-24)
- Each layer contains:
  - Multi-head self-attention
  - Feed-forward network
  - Layer normalization
  - Residual connections

### Decoder Stack
- Similar to encoder but with modifications:
  - Masked self-attention (prevents looking at future tokens)
  - Encoder-decoder cross-attention
  - Standard feed-forward network

### Full Architecture
```
Input Embeddings + Positional Encodings
         ↓
   [Encoder Stack]
   - Layer 1, 2, ..., N
         ↓
   [Decoder Stack]
   - Layer 1, 2, ..., N
   (with cross-attention to encoder)
         ↓
   Linear + Softmax
         ↓
   Output Probabilities
```

## Mathematical Details

### Scaled Dot-Product Attention

**Motivation for Scaling**:
- Large dimension d_k causes dot products to grow large
- Large values push softmax to extreme regions (small gradients)
- Scaling by sqrt(d_k) stabilizes training

**Complexity**: O(n^2 * d_k) where n = sequence length

### Multi-Head Attention Benefits

1. **Representation subspace**: Different heads learn different patterns
   - Syntax, semantics, long-range dependencies
   - Position-specific information

2. **Robustness**: Averaging over multiple attention patterns

3. **Parallelization**: Heads computed independently

## Advantages Over RNNs

| Feature | RNNs | Transformers |
|---------|------|---------------|
| Parallelization | Sequential (slow) | Full parallelization (fast) |
| Long-range dependencies | Vanishing gradient | Direct connections |
| Memory usage | O(n) | O(n²) attention |
| Training speed | Slow | Fast |
| Maximum length | Limited by gradient | Limited by memory |
| Inference latency | Fast (autoregressive) | Must decode sequentially |

## Key Design Choices

### 1. Residual Connections
- Enables very deep networks
- Facilitates gradient flow
- Formula: x' = Layer(x) + x

### 2. Dropout
- Applied to attention weights and FFN outputs
- Prevents overfitting
- Typical rate: 0.1

### 3. Embedding Scaling
- Embeddings scaled by sqrt(d_model)
- Prevents embeddings from dominating positional encodings

### 4. Initialization
- Xavier/Glorot initialization
- Careful weight initialization for stability

## Capacity and Scaling

### Model Sizes
- **Base**: 110M parameters (BERT-base)
- **Large**: 340M parameters (BERT-large)
- **XL**: 1.3B+ parameters (GPT-2, GPT-3)
- **Massive**: 175B+ (GPT-3)

### Scaling Laws
- Performance improves with model size
- Performance improves with data size
- Compute-optimal model size grows with data

## Computational Complexity

**Per-layer complexity**:
- Self-attention: O(n² × d_model)
- Feed-forward: O(n × d_ff)
- Total per layer: O(n² × d_model) for long sequences

**Optimizations**:
- Linear attention approximations
- Sparse attention patterns
- Hierarchical representations

## Pre-training and Fine-tuning

### Pre-training Objectives
- **Masked Language Modeling**: Predict masked tokens
- **Causal Language Modeling**: Predict next token
- **Contrastive Learning**: Distinguish similar vs dissimilar

### Transfer Learning
- Pre-trained models capture general language knowledge
- Fine-tuning adapts to specific tasks
- Requires fewer labeled examples

## Learning Objectives

After this section, understand:
- Self-attention mechanism and why it's powerful
- Multi-head attention and its benefits
- Transformer encoder-decoder architecture
- Positional encoding and its role
- Advantages over sequential models
- Scaling properties of Transformers
- Trade-offs between performance and efficiency
