# SVM Fundamentals

## Overview
Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that maximizes the margin between different classes.

## Key Concepts

### 1. Hyperplane
- A decision boundary that separates different classes
- In 2D: a line, in 3D: a plane, in n-dimensions: a hyperplane
- Mathematical form: w·x + b = 0
- w: weight vector (perpendicular to hyperplane)
- b: bias term

### 2. Margin
- The distance between the hyperplane and the nearest data points
- Wider margins lead to better generalization
- Support vectors are the data points closest to the hyperplane

### 3. Support Vectors
- The data points that lie on or violate the margin
- These points determine the position and orientation of the hyperplane
- Only these points matter for the final decision boundary
- Removing non-support vectors doesn't change the model

## Linear vs Non-linear SVM

### Linear SVM
- Used when data is linearly separable
- Finds a straight line/plane to separate classes
- Fast and interpretable
- Limited to simple problems

### Non-linear SVM
- Uses kernel trick to handle non-linearly separable data
- Maps data to higher-dimensional space
- Can capture complex decision boundaries
- More computationally expensive

## The Optimization Problem

For linear SVM with hard margin (separable data):
```
Minimize: ||w||²/2
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i
```

Where:
- yᵢ ∈ {-1, +1} is the class label
- xᵢ is the feature vector
- We want to maximize margin (1/||w||) by minimizing ||w||

## Soft Margin SVM

For real-world non-separable data:
```
Minimize: ||w||²/2 + C·Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ and ξᵢ ≥ 0
```

Where:
- ξᵢ (xi) are slack variables allowing misclassification
- C is regularization parameter controlling margin violations
- Large C: smaller margin, less misclassification
- Small C: larger margin, more misclassification

## Advantages of SVM

1. **Effective in high dimensions**: Works well with many features
2. **Memory efficient**: Only uses support vectors
3. **Versatile**: Can use different kernel functions
4. **Robust**: Good generalization with proper parameter tuning
5. **Theoretical foundation**: Strong mathematical basis

## Disadvantages of SVM

1. **Computational complexity**: O(n²) to O(n³) training time
2. **Hyperparameter tuning**: C and gamma require careful tuning
3. **Scalability issues**: Slow on very large datasets
4. **Black box**: Difficult to interpret results
5. **Sensitive to feature scaling**: Requires normalization

## When to Use SVM

- Binary classification problems
- High-dimensional data
- When you have limited training data
- When interpretability is less important
- Text classification and image classification
- Medical diagnosis systems

## Common Applications

1. **Text Classification**: Spam detection, sentiment analysis
2. **Image Recognition**: Face detection, handwriting recognition
3. **Medical Diagnosis**: Disease prediction, patient classification
4. **Financial Prediction**: Stock price prediction, credit risk
5. **Bioinformatics**: Protein structure prediction

## References
- Vapnik, V. (1995). The Nature of Statistical Learning Theory
- Cortes, C., & Vapnik, V. (1995). Support-vector networks
- Schölkopf & Smola (2002). Learning with Kernels
