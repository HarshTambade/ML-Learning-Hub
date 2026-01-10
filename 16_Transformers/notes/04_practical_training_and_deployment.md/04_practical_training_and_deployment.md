# Practical Training and Deployment

## Training Best Practices

### Data Preparation
- Tokenization (WordPiece, BPE, SentencePiece)
- Padding and attention masks
- Batch creation with dynamic padding
- Train/val/test split (80/10/10)

### Hyperparameters
- Learning rate: 1e-5 to 5e-5
- Batch size: 32-64 (fine-tuning)
- Epochs: 3-5 for fine-tuning
- Optimizer: Adam with weight decay
- Warmup: 10% of steps

### Training Tips
1. Use mixed precision (FP16)
2. Gradient accumulation for large batches
3. Gradient clipping (norm=1.0)
4. Early stopping on validation loss
5. Learning rate scheduler (linear, cosine)

## Evaluation Metrics

### NLP Tasks
- **Classification**: Accuracy, F1, Precision, Recall
- **NER**: Micro/Macro F1, Precision-Recall
- **QA**: Exact Match, F1 score
- **Translation**: BLEU, METEOR, TER
- **Summarization**: ROUGE-1/2/L, BLEURT

### Benchmarks
- GLUE: 9 classification tasks
- SuperGLUE: Advanced NLU
- SQuAD: Question answering
- MMLU: Multi-task language understanding

## Deployment

### Model Export
- PyTorch to ONNX
- HuggingFace transformers format
- TensorFlow conversion
- Quantized versions (INT8)

### Serving Options
- HuggingFace Model Hub
- TorchServe
- TensorFlow Serving
- Replicate API
- Together.ai
- AWS SageMaker
- Google Cloud AI Platform

### Inference Optimization
- Batch processing
- Quantization (4-bit, 8-bit)
- Pruning
- Knowledge distillation
- GPU/CPU selection

## Monitoring

### Metrics
- Model performance drift
- Inference latency
- Resource utilization
- Error rates

### Logging
- Request/response logs
- Model predictions
- Feature values
- Performance metrics

## Common Pitfalls

1. **Overfitting**: Too many epochs, small dataset
2. **Class Imbalance**: Weighted sampling/loss
3. **Out-of-vocabulary**: Special tokens handling
4. **Position Limits**: Truncating long sequences
5. **Batch Size**: Too large causes OOM

## Learning Objectives
- Configure effective training pipelines
- Understand evaluation metrics
- Deploy models to production
- Monitor model performance
- Optimize inference speed
