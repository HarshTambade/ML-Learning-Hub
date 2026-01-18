# Neural Networks for Time Series

## RNN (Recurrent Neural Networks)
**Architecture**: Processes sequences with hidden state
- Input → Hidden (uses previous hidden state) → Output
- Captures long-range dependencies
- Suffers from vanishing gradient problem

## LSTM (Long Short-Term Memory)
**Improvements over RNN**:
- Cell state preserves information
- Input gate: What to remember
- Forget gate: What to discard
- Output gate: What to output
- Solves vanishing gradient problem

## GRU (Gated Recurrent Unit)
- Simpler than LSTM
- Fewer parameters
- Similar performance
- Reset gate & Update gate

## Sequence-to-Sequence Architecture
- Encoder: Reads input sequence
- Decoder: Generates output sequence
- Attention mechanism: Focus on relevant parts
- Used for: Multi-step forecasting

## Advantages
- Captures non-linear patterns
- Handles long sequences
- No stationarity assumption
- Can learn complex relationships

## Disadvantages
- Requires large datasets
- Longer training time
- Hyperparameter tuning crucial
- Black box nature

## Best Practices
- Scale data properly
- Use validation set for tuning
- Early stopping to prevent overfitting
- Check for overfitting on test set
