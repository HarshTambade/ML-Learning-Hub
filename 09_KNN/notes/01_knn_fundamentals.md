# KNN Fundamentals

## Core Concept
K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies points based on the majority class of their K nearest neighbors in the training set.

## How It Works
1. Store all training data
2. For each query point:
   - Calculate distance to all training points
   - Find K nearest neighbors
   - Majority vote (classification) or average (regression)

## Key Characteristics
- **Lazy Learner**: No model training, all computation happens at prediction time
- **Non-parametric**: Makes no assumptions about data distribution
- **Instance-based**: Relies on actual data points, not learned parameters
- **Local Methods**: Decision boundary depends on nearby points only

## Distance Metrics
- **Euclidean**: √(Σ(xi-yi)²) - Default, works for most continuous data
- **Manhattan**: Σ|xi-yi| - Better for grid-like data
- **Minkowski**: (Σ|xi-yi|^p)^(1/p) - Generalizes Euclidean and Manhattan
- **Cosine**: For high-dimensional sparse data (text, images)

## Choosing K
- Small K: Sensitive to noise, high variance
- Large K: Smooth boundary, high bias
- Rule of thumb: K = √n where n is training size
- Odd K: Avoids ties in binary classification
- Use cross-validation to find optimal K

## Complexity Analysis
- **Training**: O(1) - Just store data
- **Prediction**: O(n*d) - Distance to all n points in d dimensions
- **Space**: O(n*d) - Store entire dataset

## Advantages
1. Simple to understand and implement
2. Works with any data distribution
3. Effective for multi-class problems
4. No explicit training phase
5. Interpretable - can examine neighbors

## Disadvantages
1. Slow prediction (O(n)) vs O(1) for trained models
2. Memory intensive - stores all data
3. Sensitive to irrelevant features
4. Requires feature scaling
5. Poor in high dimensions (curse of dimensionality)

## Real-World Applications
- Recommendation systems
- Image recognition
- Medical diagnosis
- Anomaly detection
- Credit risk assessment

## Common Issues and Solutions
| Problem | Solution |
|---------|----------|
| Imbalanced data | Use weighted KNN, SMOTE, or class weights |
| Different scales | Always scale features (StandardScaler) |
| High dimensions | Feature selection/reduction |
| Slow prediction | Use KD-trees, Ball-trees, or LSH |
| Noise sensitivity | Increase K value |

## When to Use KNN
✅ Good for: Small datasets, non-linear patterns, interpretability needed
❌ Avoid: Large datasets, high-dimensional data, real-time predictions needed
