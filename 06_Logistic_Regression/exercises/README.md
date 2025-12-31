# Logistic Regression Exercises

## Exercise 1: Sigmoid Function Implementation

**Problem**: Implement the sigmoid function and visualize it.

```python
def sigmoid(z):
    """Sigmoid function: 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))
```

**Tasks**:
- Plot sigmoid for z in range [-10, 10]
- Find value at z=0
- Explain asymptotic behavior

---

## Exercise 2: Cost Function Calculation

**Problem**: Implement and calculate binary cross-entropy loss.

**Tasks**:
- Calculate log loss for given predictions
- Understand impact of confident wrong predictions
- Compare losses for different prediction probabilities

---

## Exercise 3: Gradient Descent from Scratch

**Problem**: Implement logistic regression with manual gradient descent.

**Tasks**:
- Initialize weights
- Implement forward pass (sigmoid)
- Calculate gradients
- Update weights iteratively
- Track loss over epochs

---

## Exercise 4: Feature Scaling Impact

**Problem**: Compare model performance with/without scaling.

**Tasks**:
- Train on unscaled features
- Train on scaled features (StandardScaler)
- Compare convergence speed
- Measure final accuracy

---

## Exercise 5: Regularization Effects

**Problem**: Analyze L1 vs L2 regularization.

**Tasks**:
- Train models with different C values
- Plot coefficient magnitudes
- Count non-zero coefficients (L1 sparsity)
- Compare training vs test accuracy

---

## Exercise 6: Decision Boundary Visualization

**Problem**: Visualize decision boundaries for different models.

**Tasks**:
- Create 2D synthetic dataset
- Train logistic regression
- Plot decision boundary
- Show support/misclassified points

---

## Exercise 7: Multiclass Classification

**Problem**: Implement one-vs-rest multiclass classification.

**Tasks**:
- Load iris or wine dataset
- Train 3 binary classifiers
- Implement prediction logic
- Calculate per-class accuracies

---

## Exercise 8: ROC Curve and AUC

**Problem**: Generate ROC curve and calculate AUC.

**Tasks**:
- Vary decision threshold
- Calculate TPR and FPR
- Plot ROC curve
- Interpret AUC score

---

## Exercise 9: Cross-Validation

**Problem**: Implement k-fold cross-validation.

**Tasks**:
- Split data into k folds
- Train on k-1 folds
- Evaluate on held-out fold
- Report mean and std accuracy

---

## Exercise 10: Hyperparameter Tuning

**Problem**: Use GridSearchCV for parameter optimization.

**Tasks**:
- Define parameter grid
- Run grid search
- Report best parameters
- Compare with baseline model
- Plot accuracy surface

---

## Solutions

Refer to code_examples/ folder for complete implementations of these exercises.

---

## YouTube Resources for Each Exercise

### Exercise 1: Sigmoid Function
- **StatQuest - Sigmoid/Logistic Function**: https://www.youtube.com/watch?v=BYVeaYMps6s
- **3Blue1Brown - Neural Networks**: https://www.youtube.com/watch?v=aircArM63ks

### Exercise 2: Cost Function
- **StatQuest - Cross Entropy**: https://www.youtube.com/watch?v=6ArSQqfAYL4
- **Andrew Ng - Cost Function**: https://www.youtube.com/watch?v=0nnUjsnMWeY

### Exercise 3: Gradient Descent
- **StatQuest - Gradient Descent**: https://www.youtube.com/watch?v=sDv4f4s2SB8
- **Andrew Ng - Gradient Descent**: https://www.youtube.com/watch?v=X1E7I7_r-9g

### Exercise 4: Feature Scaling
- **Deeplearning.AI - Feature Scaling**: https://www.youtube.com/watch?v=bvR5m0nKUKU
- **Andrew Ng - Scaling Features**: https://www.youtube.com/watch?v=l3MxXZFoX1c

### Exercise 5: Regularization
- **StatQuest - Regularization**: https://www.youtube.com/watch?v=QL3gZajWfAY
- **Andrew Ng - Regularization**: https://www.youtube.com/watch?v=gBBkfCRKf4k

### Exercise 6: Decision Boundaries
- **StatQuest - Decision Trees**: https://www.youtube.com/watch?v=7VeUAPqLfQI
- **Jeremy Howard - Decision Boundaries**: https://www.youtube.com/watch?v=e2Uh7Rjv5YU

### Exercise 7: Multiclass Classification
- **StatQuest - Multi-class Logistic Regression**: https://www.youtube.com/watch?v=t0tI3S3cHeI
- **Andrew Ng - Softmax**: https://www.youtube.com/watch?v=hVkXRptsvb8

### Exercise 8: ROC Curves
- **StatQuest - ROC Curves**: https://www.youtube.com/watch?v=4jRBRDbJemM
- **Gopal Malakar - ROC AUC**: https://www.youtube.com/watch?v=4sDD3Z3bwKc

### Exercise 9: Cross-Validation
- **StatQuest - Cross Validation**: https://www.youtube.com/watch?v=fSytzGwwrhI
- **Gopal Malakar - K-Fold CV**: https://www.youtube.com/watch?v=b0VfKvJ45eQ

### Exercise 10: Hyperparameter Tuning
- **StatQuest - GridSearchCV**: https://www.youtube.com/watch?v=6dbrR-WymwM
- **Jeremy Howard - Hyperparameter Optimization**: https://www.youtube.com/watch?v=0ysqq24a1Jw

---

## Learning Path

**Recommended Order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

**Time Estimate**: 10-15 hours for all exercises

**Difficulty**: ⭐⭐⭐ Intermediate

Refer to code_examples/ folder for complete implementations of these exercises.
