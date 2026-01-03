# Distance Metrics in KNN

## 1. Euclidean Distance
- Most common distance metric
- Formula: d(p,q) = sqrt(sum((pi - qi)^2))
- Assumes continuous features
- Sensitive to feature scaling
- Appropriate for multidimensional continuous data
- Computationally efficient in low-dimensional spaces
- Performance degrades in high dimensions (curse of dimensionality)

## 2. Manhattan Distance (L1)
- Also called taxicab or rectilinear distance
- Formula: d(p,q) = sum(|pi - qi|)
- Less sensitive to outliers than Euclidean
- Better for categorical ordinal data
- Useful in grid-like structures
- More robust to high-dimensional data

## 3. Minkowski Distance
- Generalization of Euclidean and Manhattan
- Formula: d(p,q) = (sum(|pi - qi|^p))^(1/p)
- When p=1: Manhattan distance
- When p=2: Euclidean distance
- When p=infinity: Chebyshev distance

## 4. Hamming Distance
- Used for categorical/binary features
- Counts number of positions where features differ
- Formula: d(p,q) = count(pi != qi)
- Perfect for discrete data
- Common in text and DNA sequence analysis

## 5. Cosine Similarity
- Measures angle between vectors
- Formula: similarity = (p.q) / (||p|| * ||q||)
- Range: -1 to 1
- Effective for high-dimensional sparse data
- Common in text mining and NLP
- Distance = 1 - similarity

## Feature Scaling Importance
- Different metrics require appropriate scaling
- StandardScaler: subtract mean, divide by std
- MinMaxScaler: scale to [0,1] range
- Normalization needed when features have different units
- Prevents features with larger scales from dominating

## Distance Metric Selection
- Euclidean: general continuous numerical data
- Manhattan: robust features with outliers
- Hamming: categorical/binary data
- Cosine: text, high-dimensional sparse data
- Domain knowledge guides selection
