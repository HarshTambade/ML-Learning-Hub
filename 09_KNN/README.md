# Chapter 09: K-Nearest Neighbors (KNN)

## 🎯 Introduction

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms. It's a non-parametric, instance-based learning algorithm that works by finding the K nearest data points in the feature space and using their labels to make predictions.

**Key Idea**: "Tell me who your neighbors are, and I'll tell you who you are."

## 📊 KNN Decision Boundary

```
┌─────────────────────────────────────┐
│  Query Point (?)                    │
│       ↓                              │
│  Find K=3 nearest neighbors         │
│       ↓                              │
│  [Red, Red, Blue]                   │
│       ↓                              │
│  Majority Vote: RED (2 > 1)         │
│       ↓                              │
│  Classification: RED                │
└─────────────────────────────────────┘
```

## Key Characteristics

| Feature | Description |
|---------|-------------|
| **Type** | Non-parametric, lazy learner |
| **Training** | No explicit training phase - memorizes data |
| **Prediction** | Compute distance to all training points |
| **Complexity** | O(n) prediction time, O(1) training |
| **Best For** | Small to medium datasets (< 100K samples) |
| **Scalability** | Poor for high dimensions (curse of dimensionality) |

## ✅ Advantages

1. **Simple & Intuitive**: Easy to understand and implement
2. **No Assumptions**: Works with any data distribution
3. **Effective for Non-linear Data**: Captures complex patterns
4. **Multi-class Support**: Naturally handles multiple classes
5. **Lazy Learning**: Can adapt to new data easily
6. **Interpretable**: Can explain predictions by showing neighbors

## ❌ Disadvantages

1. **Computationally Expensive**: O(n) time per prediction
2. **Memory Intensive**: Stores entire training dataset
3. **Curse of Dimensionality**: Performance degrades with high dimensions
4. **Sensitive to Feature Scaling**: Different scales affect distances
5. **Imbalanced Data Issues**: Majority class can dominate
6. **Hyperparameter Tuning**: K value significantly affects performance

## How KNN Works

### Step 1: Distance Calculation

**Euclidean Distance** (most common):
```
d(A, B) = √[(x₁-x₂)² + (y₁-y₂)² + ... + (xₙ-xₙ)²]
```

**Manhattan Distance**:
```
d(A, B) = |x₁-x₂| + |y₁-y₂| + ... + |xₙ-xₙ|
```

**Minkowski Distance**:
```
d(A, B) = (|x₁-x₂|ᵖ + |y₁-y₂|ᵖ + ... + |xₙ-xₙ|ᵖ)^(1/p)
```

### Step 2: Find K Nearest Neighbors

1. Calculate distance from query point to all training points
2. Sort by distance
3. Select K nearest points

### Step 3: Make Prediction

**Classification**: Majority vote among K neighbors
**Regression**: Average of K neighbor values

## Key Parameters

### K (Number of Neighbors)

**Effect**:
- **Small K (e.g., 1)**: Sensitive to noise, may overfit
- **Large K (e.g., n)**: Smooth boundary, may underfit
- **Typical Range**: 3 to 10 for most datasets

**Formula**: Often use K = √n where n is training set size

### Distance Metric

```python
metric = 'euclidean'   # Most common
metric = 'manhattan'   # For grid-like data
metric = 'minkowski'   # General form
metric = 'cosine'      # For text/high-dimensional data
```

### Weights

```python
weights = 'uniform'    # All neighbors equally important
weights = 'distance'   # Closer neighbors weighted more
```

## Real-World Applications

- **Recommendation Systems**: Netflix, Amazon recommendations
- **Image Recognition**: Facial recognition, object detection
- **Medical Diagnosis**: Disease classification from patient data
- **Credit Scoring**: Risk assessment for loan approval
- **Anomaly Detection**: Fraud detection in financial transactions
- **Text Classification**: Spam detection, sentiment analysis

## Quick Start Code

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale features (IMPORTANT for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## Chapter Contents

### 📁 code_examples/

Practical code implementations:

1. **01_basic_knn_classification.py** - Iris dataset with KNN
2. **02_distance_metrics_comparison.py** - Euclidean vs Manhattan vs Minkowski
3. **03_knn_regression.py** - KNN for regression problems
4. **04_optimal_k_selection.py** - Finding best K value
5. **05_knn_weighted.py** - Distance-weighted KNN

### 📚 notes/

Detailed theoretical notes:

1. **01_knn_fundamentals.md** - Core concepts and theory
2. **02_distance_metrics.md** - Distance calculations and metrics
3. **03_hyperparameter_tuning.md** - K selection and optimization
4. **04_practical_applications.md** - Real-world use cases
5. **05_troubleshooting_and_tips.md** - Common issues and solutions

### 🏋️ exercises/

Practice problems and challenges:

- Exercise 1: Basic KNN on Iris (Beginner)
- Exercise 2: Distance metric comparison (Beginner-Intermediate)
- Exercise 3: Finding optimal K (Intermediate)
- Exercise 4: KNN regression (Intermediate)
- Exercise 5: Real-world dataset (Intermediate-Advanced)

### 🚀 projects/

Real-world project ideas:

- **Text Recommendation System** - Content-based recommendations
- **Image Classification** - Handwritten digit recognition
- **Medical Diagnosis** - Disease prediction from symptoms
- **Anomaly Detection** - Fraud detection in transactions
- **Customer Segmentation** - Market basket analysis

## Learning Path

### 🟢 Beginner (Hours 1-2)

- [ ] Understand basic KNN concept
- [ ] Load and split dataset
- [ ] Train simple KNN classifier
- [ ] Make predictions
- [ ] Calculate accuracy

### 🟡 Intermediate (Hours 3-6)

- [ ] Explore distance metrics
- [ ] Scale features properly
- [ ] Find optimal K value
- [ ] Evaluate with cross-validation
- [ ] Handle imbalanced data

### 🔴 Advanced (Hours 7-12)

- [ ] Implement KNN from scratch
- [ ] Use weighted KNN
- [ ] Apply to real datasets
- [ ] Optimize for large datasets
- [ ] Handle high-dimensional data

## Key Concepts at a Glance

| Concept | Definition | Impact |
|---------|-----------|--------|
| **K Value** | Number of neighbors to consider | Large K = smooth, Small K = sensitive |
| **Distance Metric** | How to measure similarity | Different metrics for different data |
| **Feature Scaling** | Normalize feature ranges | Essential for distance-based algorithms |
| **Curse of Dimensionality** | High dimensions hurt performance | Feature selection/reduction important |
| **Lazy Learning** | No model building phase | Fast training, slow prediction |

## Performance Metrics

**Classification**:
- Accuracy
- Precision & Recall
- F1 Score
- ROC-AUC

**Regression**:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## Resource Links

### Recommended YouTube Channels

- [StatQuest with Josh Starmer](https://www.youtube.com/watch?v=MDniRwXizWo) - KNN explanation
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) - Mathematics behind ML
- [Krish Naik](https://www.youtube.com/@krishnaik06) - KNN tutorial

### Online Courses

- [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)
- [Fast.ai](https://course.fast.ai/) - Practical deep learning
- [Kaggle Learn](https://www.kaggle.com/learn) - Micro courses

### Books

- "Introduction to Statistical Learning" - Hastie, Tibshirani, James
- "Hands-On Machine Learning" - Aurélien Géron
- "Pattern Recognition and Machine Learning" - Christopher Bishop

## Datasets for Practice

- **Iris**: Built-in, simple classification
- **Digits**: MNIST, image classification
- **Wine**: UCI ML Repository, classification
- **Breast Cancer**: Binary classification
- **Customer Churn**: Kaggle, practical problem

## Chapter Statistics

- **Difficulty Level**: ⭐⭐ (Easy to understand)
- **Computational Cost**: ⭐⭐⭐ (High for prediction)
- **Scalability**: ⭐⭐ (Poor for large datasets)
- **Interpretability**: ⭐⭐⭐⭐⭐ (Very interpretable)
- **Practical Use**: ⭐⭐⭐⭐ (Widely used)

## Next Steps

1. Master KNN fundamentals
2. Explore different distance metrics
3. Practice on multiple datasets
4. Optimize K and other parameters
5. Move to ensemble methods (Decision Trees, Random Forests)

## Tips for Success

1. **Always Scale Features**: Distance-based algorithms need proper scaling
2. **Start with K=3**: Good default for most problems
3. **Use Odd K**: Avoids ties in binary classification
4. **Cross-Validate**: Essential for finding optimal K
5. **Check Class Balance**: Imbalanced data affects voting
6. **Test Different Metrics**: Distance metric matters
7. **Profile Performance**: Identify bottlenecks
8. **Document Decisions**: Track K selection rationale

## Common Pitfalls to Avoid

- ❌ Using raw, unscaled features
- ❌ Not splitting train/test data
- ❌ Testing multiple K values on test set
- ❌ Ignoring class imbalance
- ❌ Using KNN on high-dimensional data without dimensionality reduction
- ❌ Not validating with cross-validation
- ❌ Using uniform weights for imbalanced data

## Last Updated

**Date**: January 3, 2026  
**Version**: 1.0  
**Author**: ML-Learning-Hub Contributors

---

**Happy Learning!** 🚀📚  
Dive into the code examples and start mastering KNN today!
