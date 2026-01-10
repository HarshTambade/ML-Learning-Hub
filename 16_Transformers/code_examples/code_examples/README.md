# Transformer Code Examples

Practical implementations of transformer architectures using PyTorch and Hugging Face Transformers library.

## Files Overview

### 1. Basic Transformer (`01_basic_transformer.py`)
- Complete transformer architecture from scratch
- Multi-head attention mechanism
- Feed-forward networks
- Positional encoding
- Layer normalization and residual connections

### 2. BERT Implementation (`02_bert_implementation.py`)
- BERT-based text classification
- Using Hugging Face pretrained models
- Fine-tuning for downstream tasks
- Attention masking

### 3. GPT Language Model (`03_gpt_language_model.py`)
- Text generation using GPT-2
- Sampling with temperature control
- Token encoding/decoding

## Installation

```bash
pip install torch transformers numpy
```

## Usage Examples

```python
# Basic Transformer
from transformer import Transformer
model = Transformer(vocab_size=10000, d_model=512, num_heads=8, d_ff=2048, num_layers=6)

# BERT Classifier
from bert_implementation import BERTClassifier
model = BERTClassifier(num_classes=2)

# GPT Generator
from gpt_language_model import GPT2TextGenerator
generator = GPT2TextGenerator()
text = generator.generate("The future is")
```

## Key Concepts

- **Multi-Head Attention**: Allows the model to focus on different representation spaces
- **Positional Encoding**: Encodes position information in the sequence
- **Self-Attention**: Each element attends to all elements in the sequence
- **Pre-training**: Models trained on large corpora before fine-tuning

## References

- Attention is All You Need (Vaswani et al., 2017)
- BERT: Pre-training of Deep Bidirectional Transformers
- Language Models are Unsupervised Multitask Learners (GPT-2)
