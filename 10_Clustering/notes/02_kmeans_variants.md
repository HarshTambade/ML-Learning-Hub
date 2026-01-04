# 02: K-Means and Advanced Variants

## Overview

K-Means is one of the most popular and simplest clustering algorithms. It partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids. This note covers the algorithm, variants, mathematical foundations, and practical considerations.

## K-Means Algorithm

### Mathematical Formulation

**Objective Function (Minimizing Within-Cluster Variance):**
```
J = Σ_{k=1}^{K} Σ_{i: x_i ∈ C_k} ||x_i - μ_k||^2
```
Where:
- K: Number of clusters
- C_k: Set of points assigned to cluster k
- μ_k: Centroid of cluster k
- x_i: Data point

### Algorithm Steps

1. **Initialization**
   - Randomly select K initial centroids from data
   - OR use K-Means++ initialization (see below)

2. **Assignment Step**
   ```
   C_k = {x_i : argmin_j ||x_i - μ_j||^2}
   ```
   - Assign each point to nearest centroid
   - Complexity: O(n*K*d)

3. **Update Step**
   ```
   μ_k = (1/|C_k|) Σ_{x_i ∈ C_k} x_i
   ```
   - Recalculate centroid for each cluster
   - Complexity: O(n*d)

4. **Convergence Check**
   - Repeat until centroids don't change significantly
   - OR maximum iterations reached
   - Convergence measured by: ||J_new - J_old|| < ε

### Convergence Guarantee

- **Guaranteed to converge** (not necessarily to global optimum)
- **Monotonically decreases** the objective function J
- **Local optimum** is reached (may not be global)
- **Convergence time** typically fast (< 20 iterations)
- **No strict convergence**: Can oscillate near optimum

## K-Means++ Initialization

### Motivation

Random initialization can lead to poor clustering and slow convergence. K-Means++ intelligently initializes centroids.

### Algorithm

```
1. Choose first centroid randomly from data
2. For i = 2 to K:
   - Calculate D(x) = min distance from x to nearest chosen centroid
   - Choose next centroid with probability ∝ D(x)^2
3. Run standard K-Means with these initial centroids
```

### Advantages

- **Better initialization**: Spreads centroids across data
- **Faster convergence**: Fewer iterations needed
- **Better results**: Higher quality clustering
- **Theoretical guarantee**: O(log K) times optimal solution

### Implementation

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init='k-means++',      # Use K-Means++
    n_init=10,             # Number of times to run
    max_iter=300,
    random_state=42
)
labels = kmeans.fit_predict(X)
inertia = kmeans.inertia_
```

## Mini-Batch K-Means

### Motivation

Standard K-Means loads entire dataset into memory. For large datasets, Mini-Batch K-Means is more efficient.

### How It Works

```
1. Randomly sample batch of b points
2. Update each centroid by:
   - Moving towards mini-batch points
   - Using exponential smoothing:
     μ_k = (1-α)*μ_k + α*μ_k_batch
     where α = (n_batch + centers_processed) / centers_seen
3. Repeat for all batches
```

### Characteristics

| Aspect | Standard K-Means | Mini-Batch K-Means |
|--------|------------------|--------------------|
| Memory | O(n*d) | O(batch_size*d) |
| Speed | Slower | Faster (10-100x) |
| Quality | Optimal | 2-3% worse, often acceptable |
| Scalability | Limited to millions | Handles billions |
| Convergence | Guaranteed | Approximate |

### Implementation

```python
from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    n_clusters=3,
    batch_size=128,        # Mini-batch size
    n_init=10,
    max_iter=300,
    random_state=42
)
labels = mbk.fit_predict(X)
```

### When to Use

- **Large datasets** (>10GB)
- **Streaming data** or online learning
- **Memory constraints**
- **Speed critical** (acceptable quality loss)
- **Incremental updates** needed

## K-Medoids (Partitioning Around Medoids - PAM)

### Key Differences from K-Means

**K-Means:**
- Uses **centroids** (computed mean point)
- May not be actual data point
- Sensitive to outliers

**K-Medoids:**
- Uses **medoids** (actual data point)
- Robust to outliers
- Can use any distance metric

### Objective Function

```
J = Σ_{k=1}^{K} Σ_{i: x_i ∈ C_k} d(x_i, m_k)
```
Where m_k is actual data point (medoid).

### Algorithm

**Build Phase:**
- Greedily select K medoids that minimize J

**Swap Phase:**
- Iteratively swap medoids with non-medoids
- Accept swap if it reduces total cost
- Repeat until no improvement

### Advantages & Limitations

**Advantages:**
- Robust to outliers and noise
- Works with arbitrary distance metrics
- Interpretable clusters (centered on actual points)
- Works on non-Euclidean data

**Limitations:**
- O(K*(n-K)^2) per iteration (much slower)
- Not suitable for large datasets
- Still requires k specification

### Implementation

```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(
    n_clusters=3,
    metric='euclidean',
    method='alternate'     # or 'pam'
)
labels = kmedoids.fit_predict(X)
```

## Fuzzy C-Means (FCM)

### Concept

Fuzzy C-Means extends K-Means by allowing **soft assignments** - each point has a degree of membership in each cluster (0 to 1).

### Objective Function

```
J = Σ_{k=1}^{K} Σ_{i=1}^{n} u_{ik}^m ||x_i - μ_k||^2
```
Where:
- u_ik: Degree of membership of point i in cluster k
- m: Fuzziness parameter (typically 2)
- Constraint: Σ_k u_ik = 1 for all i

### Algorithm

```
1. Initialize membership matrix U randomly
2. Repeat:
   a) Update centroids:
      μ_k = (Σ_i u_{ik}^m * x_i) / Σ_i u_{ik}^m
   b) Update memberships:
      u_{ik} = 1 / Σ_j (||x_i - μ_k|| / ||x_i - μ_j||)^(2/(m-1))
3. Until convergence
```

### Characteristics

| Aspect | K-Means | Fuzzy C-Means |
|--------|---------|---------------|
| Assignment | Hard (0 or 1) | Soft (0 to 1) |
| Uncertainty | None | Quantified |
| Outlier handling | Poor | Moderate |
| Computational cost | O(n*K*d*t) | O(n*K*d*t) - similar |
| Result interpretation | Cluster labels | Membership degrees |

### Advantages

- Provides membership probabilities
- Better handling of overlapping clusters
- More nuanced cluster assignment
- Works well for continuous data

### Limitations

- Requires k specification
- Still sensitive to initialization
- More hyperparameters to tune
- Computational cost similar to K-Means

### Implementation

```python
import skfuzzy as fuzz

centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T,              # Transpose for proper shape
    c=3,              # Number of clusters
    m=2,              # Fuzziness parameter
    error=0.005,
    maxiter=1000,
    init=None
)

# Get cluster labels
labels = np.argmax(u, axis=0)

# Get membership degrees
memberships = u.T
```

## Comparison of K-Means Variants

| Algorithm | Time | Memory | Robustness | Scalability | Best Use |
|-----------|------|--------|-----------|------------|----------|
| Standard K-Means | Fast | Medium | Poor | Medium | General-purpose |
| K-Means++ | Same | Same | Better | Medium | Most cases (recommended) |
| Mini-Batch | Very Fast | Low | Fair | Excellent | Large datasets |
| K-Medoids | Very Slow | Medium | Excellent | Poor | Small datasets, outliers |
| Fuzzy C-Means | Fast | Medium | Good | Medium | Soft assignments needed |

## Parameter Selection and Tuning

### Selecting K (Number of Clusters)

**1. Elbow Method:**
```python
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    inertias.append(km.inertia_)

# Plot and look for "elbow"
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()
```

**2. Silhouette Analysis:**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Higher is better
optimal_k = K_range[np.argmax(silhouette_scores)]
```

### Convergence Tuning

```python
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    max_iter=300,          # Iterations limit
    tol=1e-4,              # Convergence tolerance
    n_init=10,             # Multiple runs
    algorithm='lloyd'      # 'lloyd' or 'elkan'
)
```

## Complexity Analysis

```
Time Complexity: O(n * K * d * iterations)
Space Complexity: O(n * d)
```

Where:
- n: Number of points
- K: Number of clusters
- d: Dimensions
- iterations: Typically 10-50

## Advantages of K-Means

1. **Simple and intuitive** - Easy to understand and implement
2. **Fast convergence** - Typically converges in 10-50 iterations
3. **Scalable** - Works well for large datasets (with Mini-Batch)
4. **Versatile** - Can be adapted to different problems
5. **Well-documented** - Extensive research and implementations
6. **Interpretable** - Cluster centroids provide insight

## Limitations and Pitfalls

1. **Local optima** - Not guaranteed global optimum
2. **Requires k** - Must specify number of clusters
3. **Sensitive to initialization** - Different initializations yield different results
4. **Spherical assumption** - Biased toward spherical clusters
5. **Outlier sensitivity** - Outliers distort centroids
6. **Scale dependent** - Features must be normalized
7. **Convergence plateau** - Can get stuck in local optimum

## Best Practices

1. **Always standardize data** - Use StandardScaler or similar
2. **Use K-Means++** - Significantly better than random initialization
3. **Multiple runs** - Set n_init to 10+ for stability
4. **Validate results** - Use silhouette, Davies-Bouldin scores
5. **Visualize** - Plot clusters to catch issues
6. **Handle outliers** - Remove or use robust variant
7. **Test K range** - Try multiple K values
8. **Use appropriate metric** - Consider domain context

## Real-World Considerations

### When K-Means Works Well
- Spherical, well-separated clusters
- Medium-sized datasets
- Known or easily estimated K
- Speed critical
- Euclidean distances appropriate

### When K-Means Struggles
- Non-spherical cluster shapes
- Clusters of vastly different sizes
- Varying cluster densities
- Presence of significant outliers
- Unknown K value

## Advanced Topics

### Kernel K-Means
- Apply K-Means in transformed (kernel) space
- Handles non-convex clusters
- More computationally expensive

### Spherical K-Means
- For high-dimensional text/document data
- Uses cosine distance instead of Euclidean
- Faster for sparse data

### Weighted K-Means
- Assign weights to features or samples
- Handle imbalanced clusters
- Incorporate domain knowledge

## Resources

- Original K-Means paper: MacQueen (1967)
- K-Means++: Arthur & Vassilvitskii (2007)
- Scikit-learn documentation
- Toward Data Science articles on K-Means
- Online clustering tutorials
