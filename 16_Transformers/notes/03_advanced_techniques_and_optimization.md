# Advanced Transformer Techniques and Optimization

## Attention Optimization

### Sparse Attention
- Reduces O(n²) to O(n log n) or O(n)
- Longformer, Big Bird, Performer
- Fixed, learned, or strided patterns

### Multi-Query Attention
- Single K,V shared across query heads
- Reduces parameters and memory
- Used in Falcon, PaLM-2

## Model Compression

### Quantization
- INT8: 4x memory reduction
- FP16: 2x speedup, 50% memory
- INT4: Extreme compression

### Knowledge Distillation  
- Train small model from large
- DistilBERT: 40% smaller, 60% faster
- Preserves ~95% performance

### Pruning
- Remove unimportant weights
- 50-90% sparsity achievable
- Iterative or one-shot methods

## Efficient Training

### Gradient Checkpointing
- Recompute activations not store
- 50% memory reduction
- Modest speed cost

### Flash Attention
- Optimize I/O complexity
- 3x speedup, same quality
- Tiling-based implementation

### Mixed Precision
- FP32 master weights
- FP16 forward/backward
- 2x speedup, 50% memory

## Parameter Efficient Fine-tuning

### LoRA
- Low-rank decomposition
- 10,000x fewer parameters
- ~0.2% accuracy loss

### QLoRA
- 4-bit quantization + LoRA
- Fine-tune 70B models on GPU

## Context Extension

### RoPE
- Rotary embeddings
- Extrapolates to longer sequences
- Used in LLaMA, Falcon

### ALiBi
- Attention with Linear Biases
- Constant memory for long context

## Prompt Engineering

### Chain-of-Thought
- Step-by-step reasoning
- 30-50% accuracy improvement

### Few-shot Learning
- In-context examples
- Better than zero-shot
- Dynamic example selection

## Learning Objectives
- Attention optimization methods
- Compression and distillation
- Efficient training techniques
- Parameter-efficient fine-tuning
- Prompt engineering strategies
