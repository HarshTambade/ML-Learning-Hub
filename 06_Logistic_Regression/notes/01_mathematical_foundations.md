# Logistic Regression: Mathematical Foundations

## Overview
Logistic Regression is a fundamental classification algorithm that models the probability of an instance belonging to a particular class. Despite its name, it's a classification algorithm, not a regression algorithm.

## The Sigmoid Function

The sigmoid function transforms any input into a probability between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- z = w^T x + b (linear combination of weights and features)
- σ(z) ∈ (0, 1) represents the probability
- When σ(z) ≥ 0.5, predict class 1
- When σ(z) < 0.5, predict class 0

## Linear Decision Boundary

Logistic regression creates a LINEAR decision boundary:

```
Decision Boundary: w^T x + b = 0
```

This means logistic regression can only solve linearly separable problems.

## Cost Function (Log Loss)

The cost function for logistic regression is Binary Cross-Entropy:

```
J(w, b) = -1/m * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]
```

Where:
- m = number of training examples
- y = actual label (0 or 1)
- ŷ = predicted probability

## Optimization: Gradient Descent

Weights are updated using gradient descent:

```
dJ/dw = 1/m * X^T (ŷ - y)
dJ/db = 1/m * Σ(ŷ - y)

Update: w = w - α * dJ/dw
```

Where α is the learning rate.

## Binary vs Multiclass

### Binary Classification (OvR)
- One classifier: P(y=1|x)
- Direct probability output

### Multiclass (One-vs-Rest)
- k classifiers for k classes
- Each predicts: P(y=i|x) vs P(y≠i|x)
- Select class with highest probability

### Multinomial (Softmax)
- Single classifier with k outputs
- Softmax: P(y=i|x) = e^(z_i) / Σ(e^(z_j))

## Regularization

### L2 Regularization (Ridge)
```
J_regularized = J + λ/2m * Σ(w^2)
```
- Shrinks all coefficients
- Handles multicollinearity

### L1 Regularization (Lasso)
```
J_regularized = J + λ/m * Σ|w|
```
- Performs feature selection
- Can force coefficients to exactly 0
- Creates sparse models

## Key Assumptions

1. **Binary or Multiclass Output**: Target variable must be categorical
2. **Independence**: Observations are independent
3. **Linearity**: Log-odds are linearly related to features
4. **No Multicollinearity**: Features should not be highly correlated
5. **Large Sample Size**: Works better with sufficient data

## Advantages

✓ Simple and interpretable
✓ Fast training and prediction
✓ Works with linearly separable data
✓ Probability estimates
✓ Handles binary and multiclass problems

## Limitations

✗ Cannot solve non-linear problems
✗ Assumes linear decision boundary
✗ Sensitive to feature scaling
✗ Requires tuning regularization parameter
✗ Can be unstable with imbalanced data

## Resources

- Andrew Ng's Machine Learning Course
- Pattern Recognition and Machine Learning (PRML)
- The Elements of Statistical Learning
