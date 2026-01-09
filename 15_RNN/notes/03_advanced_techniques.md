# Advanced RNN Techniques

## Introduction

Beyond basic LSTM and GRU, advanced techniques and optimization methods are crucial for building state-of-the-art RNN systems. This section covers techniques for improving model performance, handling long sequences, and enabling more efficient training.

## Sequence-to-Sequence (Seq2Seq) Models

### Architecture Overview
Seq2Seq models process input sequences and generate output sequences of potentially different lengths. They consist of two main components:

1. **Encoder**: Processes input sequence and produces context vector
2. **Decoder**: Uses context vector to generate output sequence

### Basic Architecture
```
Input: "Hello world"
Encoder RNN: Processes word by word
Context Vector: Final hidden state from encoder
Decoder RNN: Generates output sequence using context
Output: "Bonjour le monde" (French translation)
```

### Advantages
- Handles variable-length input and output sequences
- Encoder context captures semantic meaning
- Used for translation, summarization, question answering

### Limitations
- Information bottleneck: All information squeezed into single context vector
- Struggles with very long sequences
- Loses information from early tokens in long sequences

## Attention Mechanism

### Motivation
In long sequences, the fixed-size context vector becomes a bottleneck. Attention allows the decoder to focus on different parts of the input at each decoding step.

### How Attention Works
```
1. For each decoder time step t:
   - Compute attention weights over all encoder hidden states
   - Weight = softmax(score(decoder_h_t, encoder_h_i))
   - Create context vector as weighted sum of encoder states
   - Concatenate context with decoder input
   
2. Attention score function:
   - Dot-product: score = encoder_h @ decoder_h.T
   - Additive: score = tanh(W @ [encoder_h; decoder_h])
   - Multiplicative: score = (encoder_h @ W) @ decoder_h.T
```

### Benefits
- Interpretable: Attention weights show which inputs matter
- Better long-term dependency modeling
- Handles variable-length sequences naturally
- State-of-the-art results across many tasks

## Bidirectional RNNs

### Concept
Processes sequence in both directions (forward and backward) using two RNNs, then combines results.

### Implementation
```
Forward RNN: [h_1^f, h_2^f, ..., h_n^f]
Backward RNN: [h_n^b, h_{n-1}^b, ..., h_1^b]
Output: concatenate [h_i^f; h_i^b] for each position i
```

### When to Use
- When full sequence available at inference time
- Tasks like named entity recognition, part-of-speech tagging
- Sentiment analysis on full documents
- Sequence labeling problems

## Multi-layer RNNs

### Advantages
- Deeper networks capture hierarchical patterns
- Lower layers learn local features, higher layers learn abstract patterns
- Similar to deep CNNs for hierarchical feature extraction

### Best Practices
```
Typical depths: 2-4 layers
Use residual connections: h_l = h_{l-1} + LSTM(h_{l-1})
Apply dropout between layers to prevent overfitting
Consider layer normalization for training stability
```

## Gradient-Related Challenges

### Vanishing Gradient Problem
- Gradients exponentially decrease through time steps
- Model struggles to learn long-term dependencies
- Affects vanilla RNNs more than LSTMs/GRUs

### Exploding Gradient Problem
- Gradients exponentially increase during backpropagation
- Causes numerical instability and NaN loss
- Solution: Gradient clipping

### Gradient Clipping
```
L2 clipping: if ||grad|| > threshold:
    grad = grad * threshold / ||grad||
    
Value clipping: grad = clip(grad, -threshold, threshold)
```

## Dropout and Regularization

### Standard Dropout Issues with RNNs
- Applying dropout to recurrent connections damages temporal dependencies
- Need variant that applies same mask across time steps

### Recurrent Dropout
- Apply same dropout mask at each time step
- Drop same positions in input and hidden state
- Preserves recurrent connections

### Implementation
```
for t in range(sequence_length):
    x_t = x_t * dropout_mask  # Same mask every step
    h_t = LSTM(x_t, h_{t-1})
    h_t = h_t * dropout_mask  # Same mask every step
```

## Batch Normalization and Layer Normalization

### Layer Normalization for RNNs
- Normalize across feature dimension, not batch dimension
- More stable than batch normalization for RNNs
- Applied to both input and hidden state transformations

### Benefits
- Faster convergence
- Better generalization
- More stable training

## Teacher Forcing vs Scheduled Sampling

### Teacher Forcing
- During training, use ground truth as input to next time step
- Problem: Exposure bias (mismatch between training and inference)

### Scheduled Sampling
- Gradually transition from teacher forcing to model predictions
- Begin training with teacher forcing, gradually increase prediction usage
- Reduces exposure bias

## Beam Search Decoding

### Motivation
- Greedy decoding takes best token at each step (suboptimal)
- Beam search maintains multiple hypotheses

### Algorithm
1. At each decoding step, keep top k sequences (beam width)
2. Generate next token for each sequence
3. Keep top k sequences by cumulative probability
4. Stop when all sequences reach end token

## Learning Objectives

After studying this section, you should understand:
- Seq2Seq architecture and encoder-decoder pattern
- Attention mechanisms and their benefits
- When to use bidirectional and multi-layer RNNs
- Vanishing/exploding gradient problems and solutions
- Regularization techniques specific to RNNs
- Decoding strategies for generation tasks
