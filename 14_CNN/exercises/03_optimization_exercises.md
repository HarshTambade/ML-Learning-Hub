# CNN Optimization Exercises

## Training Optimization

### Exercise 1: Learning Rate Scheduling
**Objective:** Implement and compare different learning rate schedules

**Requirements:**
- Implement StepLR scheduler
- Implement ExponentialLR scheduler
- Implement CosineAnnealingLR scheduler
- Train same model with each scheduler
- Plot and compare convergence curves

**Analysis:**
- Which scheduler converges fastest?
- Which achieves best final accuracy?
- Impact on training time

---

### Exercise 2: Optimizer Comparison
**Objective:** Compare SGD, Adam, RMSprop on CNN training

**Requirements:**
- Train model with 3 different optimizers
- Same learning rate and schedule
- Track training loss and validation accuracy
- Profile computational overhead

**Comparison Metrics:**
- Convergence speed
- Final accuracy
- Memory usage
- Gradient statistics

---

### Exercise 3: Batch Size Impact
**Objective:** Analyze effect of batch size on training

**Requirements:**
- Train with batch sizes: 16, 32, 64, 128, 256
- Keep other hyperparameters constant
- Scale learning rate appropriately
- Measure GPU memory and training time

**Analysis:**
- Accuracy vs batch size
- Training time vs batch size
- Memory efficiency
- Gradient noise effects

---

### Exercise 4: Gradient Accumulation
**Objective:** Implement gradient accumulation for larger effective batch size

**Requirements:**
- Implement accumulation with step=4
- Compare with actual batch_size=128
- Measure memory savings
- Evaluate convergence differences

---

### Exercise 5: Mixed Precision Training
**Objective:** Implement automatic mixed precision (AMP)

**Requirements:**
- Use torch.cuda.amp with autocast
- Compare float32 vs mixed precision
- Measure speedup
- Verify accuracy maintenance

---

## Regularization Tuning

### Exercise 6: Dropout Sensitivity
**Objective:** Find optimal dropout rate

**Requirements:**
- Train with dropout rates: 0, 0.1, 0.3, 0.5, 0.7
- Keep architecture same
- Plot accuracy vs dropout rate
- Analyze overfitting reduction

---

### Exercise 7: Weight Decay Analysis
**Objective:** Optimize L2 regularization strength

**Requirements:**
- Train with weight_decay: 0, 1e-5, 1e-4, 1e-3, 1e-2
- Compare train/test gap
- Monitor gradient norms
- Find best generalization

---

### Exercise 8: Ensemble Learning
**Objective:** Combine multiple models for better performance

**Requirements:**
- Train 5 models with different initializations
- Implement averaging and voting ensemble
- Evaluate ensemble accuracy
- Analyze correlation between models

---

## Inference Optimization

### Exercise 9: Model Quantization
**Objective:** Reduce model size with quantization

**Requirements:**
- Implement int8 quantization
- Measure model size reduction
- Benchmark inference speed
- Evaluate accuracy drop

---

### Exercise 10: Pruning Strategies
**Objective:** Remove redundant connections

**Requirements:**
- Implement structured and unstructured pruning
- Prune 30%, 50%, 70% of weights
- Maintain >95% accuracy
- Compare compression vs accuracy trade-off

---

### Exercise 11: Knowledge Distillation
**Objective:** Train small student model from large teacher

**Requirements:**
- Train large teacher model
- Train student model with distillation
- Compare student alone vs distilled student
- Analyze compression efficiency

---

### Exercise 12: Model Export
**Objective:** Export model for deployment

**Requirements:**
- Export to ONNX format
- Export to TorchScript
- Load and test inference
- Verify output consistency

---

## Advanced Optimization

### Exercise 13: Hyperparameter Grid Search
**Objective:** Systematic hyperparameter optimization

**Requirements:**
- Define search space
- Implement grid or random search
- Train multiple configurations
- Find best hyperparameters
- Visualize results

---

### Exercise 14: Early Stopping Implementation
**Objective:** Prevent overfitting with early stopping

**Requirements:**
- Implement patience-based early stopping
- Monitor validation loss
- Save best model checkpoint
- Compare with fixed epochs

---

### Exercise 15: Cyclical Learning Rates
**Objective:** Implement cyclic learning rate schedules

**Requirements:**
- Implement triangular schedule
- Implement cosine schedule
- Compare with constant learning rate
- Analyze training dynamics

---

## Performance Profiling

### Task 1: Training Profiling
- Use torch.profiler to analyze bottlenecks
- Identify slowest operations
- Profile memory usage
- Optimize hotspots

### Task 2: Inference Profiling
- Measure latency per layer
- Identify inference bottlenecks
- Optimize for deployment
- Compare CPU vs GPU performance

### Task 3: Memory Analysis
- Monitor peak memory usage
- Analyze memory per layer
- Implement gradient checkpointing
- Reduce memory footprint

---

## Optimization Checklist

- [ ] Choose appropriate optimizer
- [ ] Schedule learning rate
- [ ] Tune batch size
- [ ] Tune regularization
- [ ] Implement early stopping
- [ ] Profile model performance
- [ ] Optimize for inference
- [ ] Document all choices

---

## Bonus Challenges

1. **AutoML:** Implement Bayesian optimization for hyperparameter tuning
2. **Federated Learning:** Implement federated averaging
3. **Continual Learning:** Train on stream of new tasks
4. **Few-Shot Learning:** Train with limited labeled data

---

## Expected Results

- Learning rate scheduling: 5-10% accuracy improvement
- Batch size optimization: Up to 2x training speedup
- Mixed precision: 2-3x speedup with <1% accuracy drop
- Quantization: 4x model compression with ~2% accuracy drop
- Pruning: 3-5x compression at 95% accuracy retention
- Distillation: 10x compression with similar accuracy
