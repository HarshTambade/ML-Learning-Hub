# Principal Component Analysis (PCA): Theory and Applications

## PCA Overview

**Principal Component Analysis (PCA)** is a linear dimensionality reduction technique that finds new axes (principal components) where data has maximum variance.

### Key Idea
- Find directions of maximum variance in data
- Project data onto these directions
- First PC explains most variance, second PC explains next most, etc.
- Stop when we've captured sufficient variance (~95%)

## Mathematical Foundation

### Step 1: Standardize Data
```
X_scaled = (X - mean) / std
```

### Step 2: Compute Covariance Matrix
```
Cov = (1/n) * X^T * X
```

### Step 3: Eigenvalue Decomposition
```
Cov = U * Λ * U^T
```
Where:
- U = eigenvectors (principal components)
- Λ = diagonal matrix of eigenvalues (variances)

### Step 4: Select Top k Components
Choose k eigenvectors with largest eigenvalues:
```
X_transformed = X * U[:, :k]
```

## Variance Explained

**Explained Variance Ratio**:
```
var_explained[i] = λᵢ / Σλⱼ
```

**Cumulative Explained Variance**:
```
cumsum_var = Σλᵢ / Σλⱼ for i=1 to k
```

## Advantages

1. **Linear transformation** - easy to apply
2. **Interpretable** - eigenvectors show feature weights
3. **Fast computation** - O(d³) or O(n²*d) with SVD
4. **Unsupervised** - uses only data, not labels
5. **Optimal reconstruction** - minimizes MSE for given k

## Disadvantages

1. **Assumes linearity** - may miss nonlinear patterns
2. **Sensitive to scaling** - must standardize features
3. **Interpretability loss** - components are linear combinations
4. **Computational cost** - expensive for very large d
5. **Global structure only** - ignores local neighborhoods

## Variants of PCA

### 1. Incremental PCA
- Process data in batches
- Useful for streaming data
- Memory efficient

### 2. Kernel PCA
- Nonlinear dimensionality reduction
- Uses kernel trick
- Can capture curved manifolds

### 3. Sparse PCA
- Forces some loadings to zero
- More interpretable components
- Better for high-dimensional data

### 4. Probabilistic PCA
- Generative model
- Can handle missing data
- Provides uncertainty estimates

## How to Choose Number of Components

### Method 1: Variance Threshold
```
Choose k where cumsum_var >= 0.95
Usually k = 2-10 for visualization
Usually k = 0.9*d for downstream tasks
```

### Method 2: Elbow Method
- Plot explained variance vs k
- Choose k at "elbow" point
- Visual but subjective

### Method 3: Cross-validation
- Train model on k components
- Evaluate on validation set
- Choose k with best performance

### Method 4: Scree Plot
- Plot eigenvalues in decreasing order
- Choose where plot "flattens"

## Practical Considerations

1. **Always standardize** - PCA is sensitive to scale
2. **Remove outliers** - they can dominate variance
3. **Check assumptions** - PCA assumes linear relationships
4. **Validate choice** - check downstream model performance
5. **Document** - save transformation for test data

## Common Use Cases

- **Image compression** - retain 90% variance, 10x reduction
- **Visualization** - project to 2D/3D for plotting
- **Noise reduction** - discard low-variance components
- **Feature engineering** - create new features
- **Preprocessing** - improve downstream model

## Limitations

1. **Linear only** - Swiss roll, nonlinear manifolds fail
2. **Global structure** - ignores local clusters
3. **Interpretability** - hard to interpret principal components
4. **Sensitivity** - sensitive to outliers and scaling

## When to Use PCA

✓ Linear data structures
✓ Need interpretable components
✓ Computational efficiency important
✓ Visualization required
✓ Unsupervised setting

✗ Nonlinear structures
✗ Need local structure preservation
✗ Very sparse data
✗ Missing values
