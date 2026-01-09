# RNN Fundamentals

## Overview
Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to process sequential data. Unlike feedforward networks that process inputs in a single pass, RNNs maintain internal memory (hidden state) that allows them to process sequences of variable length.

## Key Concepts

### 1. Sequential Data Processing
- RNNs excel at tasks involving sequences: text, time series, speech, video
- They process data one time step at a time while maintaining a hidden state
- The hidden state acts as a "memory" carrying information from previous time steps

### 2. Recurrent Connections
- The defining characteristic is feedback connections from hidden layer to input
- At each time step t: h_t = activation(W_h * h_{t-1} + W_x * x_t + b_h)
- Hidden state is updated based on current input and previous hidden state

### 3. Parameter Sharing
- Same weights are used across all time steps
- Dramatically reduces number of parameters compared to unrolling the network
- Enables processing of variable-length sequences

## Mathematical Foundation

### RNN Cell
For each time step t:
```
x_t: Input at time step t
h_{t-1}: Hidden state from previous time step
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b_h)  # Vanilla RNN
y_t = W_y @ h_t + b_y  # Output
```

### Dimensions
- x_t: (batch_size, input_size)
- h_t: (batch_size, hidden_size)
- W_h: (hidden_size, hidden_size)
- W_x: (input_size, hidden_size)
- y_t: (batch_size, output_size)

## Advantages

1. **Memory**: Can capture long-term dependencies and temporal patterns
2. **Variable Length**: Handles sequences of different lengths naturally
3. **Parameter Efficiency**: Shared weights reduce model size
4. **Flexible Architecture**: Can be adapted for various sequence tasks
5. **Contextual Understanding**: Hidden state carries context through sequence

## Limitations

1. **Vanishing Gradient Problem**: Gradients diminish exponentially over long sequences
2. **Computational Cost**: Sequential processing prevents parallelization
3. **Limited Long-term Memory**: Vanilla RNNs struggle with very long dependencies
4. **Training Difficulty**: Harder to train than feedforward networks
5. **Exploding Gradients**: Can occur in deeper RNN architectures

## Applications

1. **Natural Language Processing**: Machine translation, sentiment analysis, text generation
2. **Time Series Forecasting**: Stock prediction, weather forecasting, traffic analysis
3. **Speech Recognition**: Converting audio to text
4. **Video Analysis**: Action recognition, video captioning
5. **Music Generation**: Composing sequences of musical notes
6. **Handwriting Recognition**: Processing temporal sequence of pen strokes
7. **Anomaly Detection**: Identifying unusual patterns in sequential data

## Types of RNN Architectures

### Many-to-One
- Multiple inputs, single output
- Example: Sentiment analysis on entire document

### One-to-Many
- Single input, multiple outputs
- Example: Image captioning (image → sequence of words)

### Many-to-Many (Encoder-Decoder)
- Sequence input, sequence output
- Example: Machine translation (source language → target language)

### Bidirectional
- Processes sequence in both directions
- Uses context from future time steps as well as past

## Common Variants

1. **LSTM (Long Short-Term Memory)**: Addresses vanishing gradient with cell state and gates
2. **GRU (Gated Recurrent Unit)**: Simplified version of LSTM with fewer parameters
3. **Bidirectional RNN**: Processes sequences forward and backward
4. **Multi-layer RNN**: Stacks multiple RNN layers for deeper processing

## Learning Objectives

After studying this section, you should understand:
- How RNNs process sequential data
- The role of hidden state and parameter sharing
- Mathematical formulation of RNN cells
- Key advantages and limitations
- Common applications and architecture types
- Why vanishing/exploding gradients are problematic
