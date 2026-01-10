# Transformer Exercises

Hands-on exercises to reinforce your understanding of transformer architectures and applications.

## Exercise Categories

### Fundamentals (Easy)
1. Understanding attention mechanism
2. Implementing positional encoding
3. Building feed-forward networks
4. Layer normalization practice

### Intermediate
1. Building multi-head attention from scratch
2. Implementing transformer blocks
3. Working with token embeddings
4. Sequence masking and padding

### Advanced
1. Fine-tuning BERT on custom dataset
2. Implementing custom attention mechanisms
3. Model evaluation and benchmarking
4. Transfer learning strategies

## How to Use

Each exercise includes:
- Clear problem statement
- Expected input/output format
- Hints for implementation
- Solution verification tests

## Getting Started

```python
# Load exercise
from exercises import Exercise
exercise = Exercise.load('01_attention_mechanism')

# Read problem
print(exercise.problem)

# Write solution
solution = your_solution_here()

# Verify
exercise.verify(solution)
```

## Learning Tips

1. Start with fundamentals before advanced exercises
2. Use the provided hints if you get stuck
3. Test your code with different input sizes
4. Compare your solution with others
5. Try implementing variations and extensions

## Progress Tracking

Track your progress:
- [ ] Fundamentals (1-4)
- [ ] Intermediate (1-4)
- [ ] Advanced (1-4)

## Common Issues

**Dimension mismatch**: Verify input/output shapes match expected format
**Gradient issues**: Use gradient clipping if needed
**Memory errors**: Reduce batch size or use gradient accumulation

## Resources

- Hugging Face Transformers Documentation
- PyTorch Official Tutorials
- Papers and research articles

## Contributing

If you create interesting exercises, please contribute!
