# Decision Boundaries in Logistic Regression

## Overview

Decision boundaries are the surfaces that separate different class regions in a feature space. In logistic regression, the decision boundary divides the feature space into regions where the model predicts different classes.

## Linear Decision Boundaries

### Definition

For binary classification with logistic regression:

```
Decision Boundary: σ(z) = 0.5
=> z = 0
=> w·x + b = 0
```

Where:
- `w` = weight vector
- `x` = feature vector
- `b` = bias term

### Example in 2D

For features x1 and x2:
```
Decision Boundary: w1*x1 + w2*x2 + b = 0
=> x2 = -(w1*x1 + b) / w2
```

This forms a **straight line** separating the two classes.

## Non-Linear Decision Boundaries

### Feature Engineering

Logistic regression produces linear boundaries in the original feature space. To create non-linear boundaries, we use feature engineering:

```python
# Example: Polynomial features for non-linear boundary
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create non-linear boundary
poly_features = PolynomialFeatures(degree=2)
model = Pipeline([
    ('poly_features', poly_features),
    ('logistic_regression', LogisticRegression())
])

model.fit(X, y)
```

## Decision Boundary Visualization

### 2D Visualization Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Create mesh for boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

# Get predictions
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary in Logistic Regression')
plt.show()
```

## Mathematical Formulation

### Probability Regions

```
Class 1 Region: P(y=1|x) > 0.5
              => σ(w·x + b) > 0.5
              => w·x + b > 0

Class 0 Region: P(y=1|x) < 0.5
              => σ(w·x + b) < 0.5
              => w·x + b) < 0
```

### Margin

The distance of a point from the decision boundary:
```
Distance = |w·x + b| / ||w||
```

Larger margin indicates higher confidence in the prediction.

## Properties of Decision Boundaries

### 1. **Linearity in Original Space**
   - Standard logistic regression produces linear boundaries
   - Works well for linearly separable data

### 2. **Flexibility Through Feature Engineering**
   - Polynomial features create non-linear boundaries
   - Interaction terms capture feature relationships

### 3. **Probabilistic Interpretation**
   - Boundary at P(y=1|x) = 0.5
   - More intuitive than hard classification

### 4. **Smooth Transition**
   - Sigmoid function ensures smooth probability transition
   - No abrupt changes in predictions

## Real-World Example: Email Classification

### Scenario
Classify emails as spam (1) or not-spam (0) using features:
- Word frequency
- Email length
- Sender reputation

### Decision Boundary
```
Decision: 0.3*word_freq + 0.5*length - 0.2*sender_rep - 1.2 = 0

If score > 0: Classify as SPAM
If score < 0: Classify as NOT-SPAM
```

## Challenges and Solutions

### Challenge 1: Overlapping Classes
**Problem**: Classes overlap in feature space
**Solution**: 
- Collect more/better features
- Use regularization to prevent overfitting
- Consider ensemble methods

### Challenge 2: Non-Separable Data
**Problem**: No line can perfectly separate classes
**Solution**:
- Add polynomial features
- Use Kernel Logistic Regression
- Increase model complexity

### Challenge 3: High-Dimensional Data
**Problem**: Cannot visualize boundaries in > 3D
**Solution**:
- Use dimensionality reduction (PCA, t-SNE)
- Analyze feature importance
- Visualize pairwise feature relationships

## Practical Implementation Tips

### 1. **Always Standardize Features**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Why? Ensures fair weight distribution across features.

### 2. **Use Regularization**
```python
model = LogisticRegression(C=0.1, penalty='l2')
```

Prevents overfitting to training data.

### 3. **Validate on Test Data**
Ensure boundaries generalize well to unseen data.

## Key Takeaways

✅ Decision boundaries in logistic regression are **linear in the original feature space**

✅ Sigmoid function maps scores to **[0, 1] probability range**

✅ Boundary occurs at **P(y=1|x) = 0.5** (score = 0)

✅ Feature engineering enables **non-linear boundaries**

✅ Visualization crucial for understanding model behavior

## YouTube Resources

### Recommended Videos:

1. **"Decision Boundaries in Machine Learning" - StatQuest**
   - Visual explanation of linear and non-linear boundaries
   - Clear examples with code
   - [Watch on YouTube](https://www.youtube.com/watch?v=Db4xXN0XkYo)

2. **"Logistic Regression Decision Boundaries" - Andrew Ng (Coursera)**
   - Mathematical foundation
   - Boundary visualization
   - [Playlist](https://www.youtube.com/watch?v=HIQlmHxI6-0)

3. **"Non-linear Decision Boundaries" - Jeremy Jordan**
   - Feature engineering techniques
   - Polynomial and interaction terms
   - [YouTube Tutorial](https://www.youtube.com/watch?v=PwAGEZ2kWuc)

4. **"Logistic Regression in scikit-learn" - Krish Naik**
   - Practical implementation
   - Boundary visualization code
   - [Complete Tutorial](https://www.youtube.com/watch?v=VCJdg7YBbAQ)

5. **"Decision Boundaries with Kernel Methods" - Victor Lavrenko**
   - Kernel tricks for non-linear boundaries
   - SVM vs Logistic Regression
   - [YouTube](https://www.youtube.com/watch?v=Qc5IyLW_hno)

## Further Reading

- James et al. "An Introduction to Statistical Learning" - Chapter on Logistic Regression
- Bishop "Pattern Recognition and Machine Learning" - Classification section
- Hastie, Tibshirani & Friedman "The Elements of Statistical Learning"

---

**Last Updated**: 2024
**Difficulty Level**: Intermediate
**Prerequisites**: Logistic regression basics, Sigmoid function understanding
