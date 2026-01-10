# BERT and Pre-trained Transformer Models

## Introduction

BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by demonstrating the power of pre-trained language models. This chapter covers major pre-trained models and their applications.

## BERT Architecture

### Overview
BERT uses only the Transformer encoder stack, not the full encoder-decoder.
- **Bidirectional**: Processes full context in both directions
- **Pre-trained**: Trained on massive unlabeled data
- **Transfer Learning**: Fine-tuned on downstream tasks

### BERT Training Objectives

**1. Masked Language Modeling (MLM)**
- Randomly mask 15% of tokens
- Predict masked tokens from context
- [MASK] token with 80% probability
- Random token with 10% probability
- Keep original with 10% probability

**2. Next Sentence Prediction (NSP)**
- Predict if second sentence follows first
- Binary classification task
- Helps model understand sentence relationships

### BERT Variants
- **BERT-base**: 12 layers, 768 hidden dim, 110M params
- **BERT-large**: 24 layers, 1024 hidden dim, 340M params
- **RoBERTa**: Improved pre-training (better NSP, longer training)
- **ALBERT**: Factorized embeddings (reduced parameters)
- **DistilBERT**: Distilled version (40% smaller, 60% faster)

## GPT Models

### GPT Evolution

**GPT-2 (1.5B parameters)**
- Causal language modeling objective
- Decoder-only architecture
- Impressive zero-shot performance

**GPT-3 (175B parameters)**
- Few-shot learning capabilities
- Emergent abilities with scale
- Versatile across tasks
- Not fine-tuned per task

**GPT-4 (Unknown architecture)**
- Multimodal (text + images)
- More reliable and safer
- Better reasoning and coding

### Key Differences: BERT vs GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Objective | Masked LM | Causal LM |
| Direction | Bidirectional | Left-to-right |
| Architecture | Encoder only | Decoder only |
| Use case | Understanding/Classification | Generation |
| Fine-tuning | Required | Optional (few-shot) |
| Training data | 3.3B words | 40GB text |

## Other Important Models

### T5 (Text-to-Text Transfer Transformer)
- Treats all tasks as text-to-text
- Single unified architecture
- Pre-trained on diverse objectives
- Sizes: Base (220M) to XXL (11B)

### ELECTRA
- Efficient learning via discriminative training
- Generator discriminator framework
- Better efficiency than BERT

### XLNet
- Autoregressive pre-training
- Permutation language modeling
- Captures bidirectional context

### BART
- Combines BERT (encoder) and GPT (decoder)
- Denoising autoencoder
- Strong for generation tasks

## Pre-training Process

### Data Preparation
- Clean and deduplicate text
- Tokenization (WordPiece, BPE, SentencePiece)
- Create training examples

### Training Details
- **Optimizer**: Adam with warmup
- **Learning rate**: Typically 1e-4
- **Batch size**: 256-512 sequences
- **Training time**: Days to weeks on TPUs/GPUs
- **Checkpointing**: Save periodically

### Computational Requirements
- BERT-base: ~4 days on 4 TPUs
- BERT-large: ~4 days on 16 TPUs
- GPT-3: ~355 GPU years

## Fine-tuning Approaches

### Standard Fine-tuning
1. Load pre-trained weights
2. Add task-specific head
3. Train on labeled data
4. Typical learning rate: 1e-5 to 5e-5

### Adapter Modules
- Small trainable bottleneck layers
- Few parameters added per task
- Reduces memory and storage

### Prompt-based Learning
- Reformulate task as language modeling
- Minimal labeled data required
- GPT-3 pioneered this approach

## Transfer Learning Effectiveness

### Why Pre-training Works
1. **General language knowledge**: Captured during pre-training
2. **Feature reuse**: Lower layers learn useful representations
3. **Reduced overfitting**: Helps with small labeled datasets
4. **Faster convergence**: Starts from good initialization

### Performance Gains
- GLUE benchmark: +5-10% improvement over non-pretrained
- Smaller datasets: +20-30% improvement
- Data efficiency: 10x fewer examples needed

## Challenges and Limitations

### Computational Cost
- Expensive to pre-train from scratch
- Requires specialized hardware
- Environmental concerns

### Bias and Ethics
- Inherit biases from training data
- May amplify harmful stereotypes
- Fairness considerations important

### Domain Adaptation
- Pre-training on general data
- May not capture domain-specific knowledge
- Domain pre-training helpful

### Interpretability
- Black box models
- Hard to explain predictions
- Attention visualization limited help

## Model Deployment

### Inference Optimization
- Quantization (INT8, FP16)
- Knowledge distillation
- Pruning and compression
- TorchScript/ONNX export

### Hosting Solutions
- Hugging Face Model Hub
- Replicate, Together.ai
- Self-hosted with TorchServe
- Cloud services (AWS, GCP, Azure)

## Learning Objectives

Understand:
- BERT architecture and pre-training objectives
- GPT models and causal vs bidirectional approaches
- Fine-tuning strategies and transfer learning
- Comparative advantages of different models
- Deployment and optimization techniques
