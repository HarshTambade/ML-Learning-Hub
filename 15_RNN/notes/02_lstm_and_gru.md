# LSTM and GRU Architectures

## Introduction

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are advanced RNN architectures designed to address the vanishing gradient problem and capture long-term dependencies in sequential data. They add gating mechanisms to control information flow.

## LSTM (Long Short-Term Memory)

### Architecture Overview
LSTM introduces a cell state (memory) and three gates to control information flow:
1. **Forget Gate**: Decides what information to discard
2. **Input Gate**: Decides what new information to store
3. **Output Gate**: Decides what to output from the cell state

### Mathematical Formulation

```
Forget Gate: f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
Input Gate: i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
Candidate Cell: C_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
Cell State: C_t = f_t * C_{t-1} + i_t * C_tilde
Output Gate: o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
Hidden State: h_t = o_t * tanh(C_t)
```

### Key Components

1. **Cell State (C_t)**
   - Carries information throughout the sequence
   - Acts as long-term memory
   - Modified by gates using element-wise multiplication and addition

2. **Forget Gate**
   - Controls which information from previous cell state to keep
   - Output range [0, 1]: 0 = forget completely, 1 = keep completely
   - Critical for learning when to reset memory

3. **Input Gate**
   - Controls which new information to add to cell state
   - Combines two components: which values to update, what new values to add
   - Prevents unnecessary updates

4. **Output Gate**
   - Controls which information from cell state to output
   - Hidden state is filtered version of cell state
   - Allows selective exposure of internal state

### Advantages

- **Long-term Dependencies**: Can learn dependencies over very long sequences
- **Gradient Flow**: Multiplicative interactions (gates) prevent vanishing gradients
- **Flexible Memory**: Cell state acts as adaptive memory that grows/shrinks as needed
- **Proven Performance**: State-of-the-art for many sequential tasks

### Disadvantages

- **Computational Cost**: More parameters and complex computations than vanilla RNN
- **Training Time**: Longer training time due to gate computations
- **Overfitting Risk**: More parameters increase overfitting risk with small datasets

## GRU (Gated Recurrent Unit)

### Architecture Overview
GRU is a simplified version of LSTM with fewer parameters. It combines forget and input gates into a single update gate.

### Mathematical Formulation

```
Reset Gate: r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)
Update Gate: z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)
Candidate Hidden: h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)
Hidden State: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
```

### Key Components

1. **Reset Gate**
   - Controls how much previous hidden state to forget
   - Used to compute candidate hidden state
   - Allows selective information from previous steps

2. **Update Gate**
   - Balances between previous hidden state and candidate hidden state
   - Replaces both forget and input gates from LSTM
   - Linear interpolation between old and new hidden state

3. **Candidate Hidden State**
   - Computed using reset gate-modified previous state
   - Similar to LSTM's cell candidate
   - Incorporates both previous state (via reset gate) and current input

### Advantages

- **Simpler Architecture**: Fewer parameters than LSTM (3 gates vs 4)
- **Faster Training**: Lower computational complexity
- **Comparable Performance**: Often performs similarly to LSTM on many tasks
- **Easier to Tune**: Fewer hyperparameters to optimize

### Disadvantages

- **Limited Capacity**: Fewer parameters may limit expressiveness on complex tasks
- **Less Control**: Merged gates provide less fine-grained control than LSTM

## LSTM vs GRU Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 gates (input, forget, output) | 2 gates (reset, update) |
| Cell State | Explicit cell state | No separate cell state |
| Parameters | ~4x more than vanilla RNN | ~3x more than vanilla RNN |
| Complexity | Higher | Lower |
| Training Speed | Slower | Faster |
| Long-term Dependency | Excellent | Good |
| Overfitting Risk | Higher | Lower |
| Dataset Size | Better for large datasets | Good for smaller datasets |

## Bidirectional RNNs/LSTMs/GRUs

### Concept
Processes sequences in both directions (forward and backward) and combines the results.

### Benefits
- Access to context from both past and future
- Better performance on tasks like named entity recognition, machine translation
- Particularly useful when full sequence is available during inference

### Architecture
```
Forward LSTM: processes sequence left-to-right
Backward LSTM: processes sequence right-to-left
Output: concatenate forward and backward hidden states
```

## Practical Considerations

### When to Use LSTM
- Complex sequential patterns requiring long-term dependencies
- Large datasets where computational cost is acceptable
- Tasks like machine translation, speech recognition
- When maximum accuracy is priority

### When to Use GRU
- Limited computational resources
- Smaller datasets (less overfitting risk)
- Similar performance to LSTM required but faster training
- Real-time applications requiring low latency

### Tips for Implementation
1. Start with GRU for quick experimentation
2. Use bidirectional variants when full sequence context available
3. Stack multiple layers for complex patterns
4. Use dropout between layers to prevent overfitting
5. Monitor validation loss for early stopping

## Learning Objectives

After studying this section, you should understand:
- How LSTM cells work and their mathematical formulation
- The role of forget, input, and output gates
- GRU architecture and simplifications vs LSTM
- When to use LSTM vs GRU
- Bidirectional variants and their applications
- Implementation considerations for real-world use
