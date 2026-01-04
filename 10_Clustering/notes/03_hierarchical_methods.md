# 03: Hierarchical Clustering Methods

## Overview
Hierarchical clustering builds a hierarchy of clusters, either by agglomerative (bottom-up) or divisive (top-down) approach. Unlike K-Means, it doesn't require specifying the number of clusters in advance and produces a dendrogram that visualizes the hierarchical structure.

## Types of Hierarchical Clustering

### 1. Agglomerative Clustering (Bottom-Up)
Starts with each data point as its own cluster and merges the closest clusters iteratively until all points are in one cluster.

**Algorithm Steps:**
1. Initialize: Each data point is a separate cluster
2. Repeat until one cluster remains:
   - Find the two closest clusters
   - Merge them into one cluster
   - Update distance matrix

**Time Complexity:** O(n²) space, O(n³) time for naive implementation

### 2. Divisive Clustering (Top-Down)
Starts with all data points in one cluster and recursively divides them into smaller clusters. Less commonly used due to computational complexity.

## Linkage Criteria

The choice of linkage criterion determines how cluster distances are calculated:

### Single Linkage
```
d(C_i, C_j) = min(d(p_i, p_j)) for p_i in C_i, p_j in C_j
```
- Distance between closest points in two clusters
- **Pros:** Computationally efficient
- **Cons:** Susceptible to chaining effect (elongated clusters)
- **Best for:** Clusters with irregular shapes

### Complete Linkage
```
d(C_i, C_j) = max(d(p_i, p_j)) for p_i in C_i, p_j in C_j
```
- Distance between farthest points in two clusters
- **Pros:** Produces more compact clusters
- **Cons:** Sensitive to outliers
- **Best for:** Spherical, well-separated clusters

### Average Linkage
```
d(C_i, C_j) = (1/(|C_i|*|C_j|)) * Σ d(p_i, p_j)
```
- Average distance between all pairs of points
- **Pros:** Good balance, less sensitive to outliers than complete
- **Cons:** Computationally more expensive than single
- **Best for:** General-purpose clustering

### Ward Linkage
```
d(C_i, C_j) = sqrt((2*|C_i|*|C_j|)/(|C_i|+|C_j|)) * ||μ_i - μ_j||
```
- Minimizes within-cluster variance
- **Pros:** Produces balanced, spherical clusters
- **Cons:** Sensitive to outliers
- **Best for:** General-purpose clustering, similar to K-Means

## Distance Metrics

Common metrics used in hierarchical clustering:

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean | √(Σ(x_i - y_i)²) | Continuous variables, spherical clusters |
| Manhattan | Σ\|x_i - y_i\| | Robust to outliers |
| Cosine | 1 - (A·B)/(\|A\|*\|B\|) | Text, high-dimensional data |
| Correlation | 1 - ρ(X,Y) | When scale matters, trend similarity |

## Dendrogram Interpretation

A dendrogram is a tree diagram showing hierarchical clustering process:

**Key Components:**
- **Leaves:** Individual data points
- **Branches:** Merged clusters
- **Height:** Distance/dissimilarity at which clusters merge
- **Cutting height:** Determines final number of clusters

**How to Cut a Dendrogram:**
1. Draw horizontal line at desired height
2. Count intersections = number of clusters
3. Higher cut = fewer clusters, lower cut = more clusters

**Determining Optimal Cut:**
- Look for largest vertical gap (significant distance between merges)
- Elbow method on dendrogram heights
- Use external validation indices

## Dendrogram Distance Thresholding

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Cut dendrogram at specific distance
clusters = fcluster(Z, t=10, criterion='distance')

# Or cut at specific number of clusters
clusters = fcluster(Z, t=3, criterion='maxclust')
```

## Scikit-learn Implementation

```python
from sklearn.cluster import AgglomerativeClustering

# Create hierarchical clustering
hac = AgglomerativeClustering(
    n_clusters=3,              # Number of clusters
    linkage='ward',            # Linkage criterion
    metric='euclidean'         # Distance metric
)

# Fit and predict
labels = hac.fit_predict(X)
```

## Advantages of Hierarchical Clustering

1. **No need to specify k in advance** - Dendrogram helps decide
2. **Meaningful interpretation** - Dendrogram shows relationships
3. **Detects hierarchical structure** - Good for nested groupings
4. **Deterministic** - Always produces same result (unlike K-Means)
5. **Flexible linkage options** - Can choose based on problem

## Limitations of Hierarchical Clustering

1. **Computational complexity** - O(n³) for most algorithms
2. **Not scalable** - Difficult for very large datasets
3. **Irreversible merges** - Mistakes in early steps cannot be corrected
4. **Dendrogram sensitivity** - Results sensitive to distance metric and linkage
5. **Memory requirements** - Stores full distance matrix (n² space)

## Complexity Analysis

| Algorithm | Time | Space |
|-----------|------|-------|
| Naive Agglomerative | O(n³) | O(n²) |
| Optimized (nearest neighbor) | O(n²) | O(n²) |
| Divisive | Exponential | O(n) |

## When to Use Hierarchical Clustering

- **When you need interpretability** - Dendrogram provides visualization
- **Hierarchical data** - Natural hierarchy in problem domain
- **Exploratory analysis** - Understand cluster structure
- **Medium-sized datasets** - Up to few thousand points
- **Mixed-type data** - Can use appropriate distance metrics
- **When k is unknown** - Helps determine optimal clusters

## Comparison with K-Means

| Aspect | Hierarchical | K-Means |
|--------|-------------|----------|
| k specification | Not needed | Required |
| Scalability | Poor | Excellent |
| Speed | O(n³) | O(nk) |
| Interpretability | Excellent | Moderate |
| Initialization | Deterministic | Random (sensitive) |
| Cluster shape | Any shape possible | Convex/spherical |
| Reversibility | No | Iterative |

## Practical Tips

1. **Standardize data** - Essential for distance-based metrics
2. **Choose linkage carefully** - Ward generally works well
3. **Use dendrogram** - Always visualize before deciding clusters
4. **Consider sample size** - Use on subsets if data too large
5. **Combine methods** - Use hierarchical to find k, then K-Means for speed
6. **Validate results** - Use silhouette, Davies-Bouldin, or Calinski-Harabasz indices

## Common Pitfalls

- **Not standardizing features** - Leads to scale bias
- **Ignoring dendrogram** - Missing important structure information
- **Wrong linkage choice** - Can produce poor results
- **Sensitive to outliers** - Some linkages more robust than others
- **Treating it as final** - Always validate with other methods

## Resources for Deeper Learning

- Scikit-learn documentation on AgglomerativeClustering
- SciPy's scipy.cluster.hierarchy module
- "Cluster Analysis" - Kaufman & Rousseeuw
- Statistical Methods for Hierarchical Clustering
