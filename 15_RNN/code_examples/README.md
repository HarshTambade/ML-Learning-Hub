# RNN Code Examples

Practical PyTorch implementations demonstrating key RNN concepts.

## Code Examples

### 1. 01_basic_lstm_text_classification.py
**Sentiment Analysis using LSTM**

- Complete dataset implementation
- LSTM model with embedding layer
- Training loop with gradient clipping
- Evaluation metrics
- Real-world sentiment classification

**Key Concepts:**
- Sequence padding and masking
- Embedding layers
- LSTM cells and hidden states
- Dropout regularization
- Classification output

**Usage:**
```python
python 01_basic_lstm_text_classification.py
```

### 2. 02_gru_bidirectional_rnn.py
**Sequence Tagging with Bidirectional GRU**

- Bidirectional RNN architecture
- Packed sequences for efficiency
- GRU cells alternative to LSTM
- Sequence-level predictions

**Key Concepts:**
- GRU gates (reset, update)
- Bidirectional processing
- Packed padded sequences
- Token-level tagging

**Usage:**
```python
python 02_gru_bidirectional_rnn.py
```

## Getting Started

1. Ensure PyTorch is installed: `pip install torch`
2. Run any example: `python <filename>.py`
3. Modify hyperparameters for experimentation
4. Study the code comments for understanding

## Architecture Patterns

### LSTM Pattern
```python
model = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=2,
    batch_first=True,
    dropout=0.3
)
```

### GRU Pattern  
```python
model = nn.GRU(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)
```

## Common Techniques

- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_`
- **Packed Sequences**: Efficient variable-length handling
- **Dropout**: Prevent overfitting
- **Embedding**: Convert tokens to vectors
- **Bidirectional**: Access context from both directions

## Next Steps

1. Extend examples with your own data
2. Implement attention mechanisms
3. Try Seq2Seq architectures
4. Experiment with different hyperparameters
5. Combine with advanced techniques from notes

## Resources

- PyTorch RNN Documentation: https://pytorch.org/docs/stable/nn.html#recurrent-layers
- Related notes in ../notes/ folder
- Exercises in ../exercises/ folder
- Projects in ../projects/ folder
