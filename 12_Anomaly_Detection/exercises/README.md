# Anomaly Detection Exercises

## Overview

These exercises will help you practice implementing anomaly detection algorithms and understand their applications.

## Exercise 1: Implement Z-Score Detection
**Difficulty**: Beginner

Implement a function that detects anomalies using the Z-score method on a 1D dataset.

**Requirements**:
- Load/generate 1D data
- Calculate Z-scores
- Set threshold at 3
- Return binary predictions
- Plot results

**Dataset Suggestion**: Use `numpy.random.randn()` with added outliers

---

## Exercise 2: Compare IQR and Z-Score
**Difficulty**: Beginner

Compare the IQR method with Z-score on multiple datasets.

**Requirements**:
- Implement both methods
- Test on normal, skewed, and mixed distributions
- Compare precision, recall, F1-score
- Visualize predictions
- Document findings

---

## Exercise 3: Tune Isolation Forest
**Difficulty**: Intermediate

Find optimal parameters for Isolation Forest on a real dataset.

**Requirements**:
- Load dataset (e.g., Credit Card Fraud)
- Implement grid search for n_estimators and contamination
- Evaluate with ROC-AUC
- Create parameter sensitivity plot
- Save best model

---

## Exercise 4: LOF Parameter Sensitivity
**Difficulty**: Intermediate

Analyze how k (n_neighbors) affects LOF performance.

**Requirements**:
- Create synthetic dataset with varying density regions
- Test multiple k values (5, 10, 20, 50)
- Measure detection accuracy
- Plot results vs. k
- Explain trade-offs

---

## Exercise 5: One-Class SVM Kernel Comparison
**Difficulty**: Intermediate

Compare different kernels in One-Class SVM.

**Requirements**:
- Test linear, rbf, and poly kernels
- Use 2D dataset for visualization
- Compare ROC-AUC scores
- Visualize decision boundaries
- Document kernel advantages/disadvantages

---

## Exercise 6: Statistical Methods on Real Data
**Difficulty**: Intermediate

Apply statistical methods to the Credit Card Fraud dataset.

**Requirements**:
- Load Kaggle credit card fraud dataset
- Apply Z-score, IQR, Mahalanobis
- Compare with Isolation Forest
- Evaluate on held-out test set
- Create comparison report

---

## Exercise 7: Autoencoder for Anomaly Detection
**Difficulty**: Advanced

Build and train an autoencoder for anomaly detection.

**Requirements**:
- Design appropriate architecture
- Train on normal data only
- Implement threshold selection
- Test on mixed data
- Visualize reconstruction errors
- Compare with classical methods

---

## Exercise 8: Multi-Algorithm Ensemble
**Difficulty**: Advanced

Combine multiple algorithms for better detection.

**Requirements**:
- Implement 3+ algorithms
- Create voting mechanism
- Tune for precision/recall tradeoff
- Evaluate ensemble performance
- Compare with individual methods
- Document why ensemble helps

---

## Solutions

Solutions will be provided after you complete the exercises. Try to solve them independently first!

---

## Tips for Success

1. **Start Simple**: Begin with beginner exercises before intermediate ones
2. **Understand Data**: Always explore your data first
3. **Baseline**: Create simple baseline before complex methods
4. **Evaluation**: Use multiple metrics, not just accuracy
5. **Visualization**: Plot your results to gain insights
6. **Documentation**: Write clear comments and explanations

---

## Common Challenges

- **Imbalanced Data**: Use appropriate metrics (PR-AUC, F1)
- **Parameter Tuning**: Use grid/random search
- **Evaluation**: Don't use accuracy on imbalanced data
- **Generalization**: Always use separate train/test sets

---

## Resources

- See `../RESOURCES.md` for learning materials
- Check `../code_examples/` for reference implementations
- Review `../notes/` for theoretical background

---

**Happy Learning!** 🚀
