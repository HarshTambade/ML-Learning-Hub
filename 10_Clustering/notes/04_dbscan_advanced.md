# 04: DBSCAN and Advanced Clustering Methods

## Overview
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters of arbitrary shape and automatically detects outliers. Unlike K-Means and hierarchical clustering, DBSCAN doesn't require specifying the number of clusters in advance and is particularly effective for discovering clusters of varying densities.

## DBSCAN Algorithm

### Core Concepts

**1. Epsilon (eps)**: The radius of the neighborhood around a point
```
N(p) = {q in D : dist(p, q) <= eps}
```

**2. MinPoints**: The minimum number of points required in the eps-neighborhood for a point to be considered a core point

**3. Point Classification**:
- **Core Point**: Has at least MinPoints points within eps distance (including itself)
- **Border Point**: Within eps distance of a core point but has fewer than MinPoints neighbors
- **Noise Point (Outlier)**: Neither core nor border point

### DBSCAN Algorithm Steps

1. For each unvisited point p:
   - Mark p as visited
   - Retrieve N(p) = all points within eps of p
   - If |N(p)| < MinPoints, mark p as noise
   - Else, create new cluster and recursively add density-reachable points
2. A point q is density-reachable from p if:
   - There's a chain of core points p=p0, p1, ..., pn=q where pi+1 in N(pi)

### Mathematical Formulation

```
Cluster C = {p in D : p is density-reachable from a core point in C}
```

**Directly Density-Reachable:**
q is directly density-reachable from p if:
- p is core point
- q in N(p)

**Density-Connected:**
p and q are density-connected if both are density-reachable from a core point r

## Time Complexity

| Aspect | Complexity | Note |
|--------|-----------|------|
| Without spatial index | O(n²) | Pairwise distance calculation |
| With spatial index (KD-tree) | O(n log n) | Efficient neighborhood search |
| Space Complexity | O(n) | Storing the dataset |

## Parameter Selection

### Choosing eps (Epsilon)

**K-distance Graph Method:**
1. Compute distance to k-th nearest neighbor for each point (k = MinPoints - 1)
2. Sort distances in descending order
3. Plot the k-distance graph
4. Look for the "elbow" point (sharp knee)
5. Use the distance at the elbow as eps

```python
from sklearn.neighbors import NearestNeighbors

# Find k-distance graph
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort distances
distances = np.sort(distances[:, -1], axis=0)

# Plot to find elbow
plt.plot(distances)
plt.ylabel('k-distance')
plt.xlabel('Data Points sorted by distance')
plt.show()
```

### Choosing MinPoints

**Heuristic:**
- Minimum rule: MinPoints >= D (where D = dimensions)
- Common practice: MinPoints = 2 * D
- Larger values for higher dimensions to avoid curse of dimensionality
- Larger values for data with high noise

```python
# General recommendation
min_points = 2 * n_features
```

## Scikit-learn Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(
    eps=0.5,              # Neighborhood radius
    min_samples=5,        # Minimum points for core point
    metric='euclidean'    # Distance metric
)

labels = dbscan.fit_predict(X_scaled)

# Identify clusters and noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')
```

## Advantages of DBSCAN

1. **Arbitrary cluster shapes** - Finds non-spherical clusters
2. **No need to specify k** - eps and min_samples are more intuitive
3. **Automatic outlier detection** - Identifies noise points (-1 label)
4. **Works with varying densities** - Compared to older density methods
5. **Deterministic** - Always produces same result (no random initialization)
6. **Efficient with spatial indexing** - Can handle large datasets

## Limitations of DBSCAN

1. **Sensitive to eps parameter** - Small changes can drastically alter results
2. **Varying density problem** - Struggles with clusters of different densities
3. **High dimensionality issues** - Distance metrics become less meaningful in high-D
4. **Border point assignment** - Border points assigned to first cluster that reaches them
5. **Parameter selection difficult** - Requires domain knowledge or extensive tuning
6. **Not deterministic with border points** - Order of processing affects border assignments

## Handling Varying Densities - OPTICS

**OPTICS (Ordering Points To Identify Clustering Structure)**:
- Extension of DBSCAN that handles varying densities
- Creates an ordering of points with reachability distances
- Can extract clusters at different densities from single run

```python
from sklearn.cluster import OPTICS

optics = OPTICS(
    min_samples=5,
    xi=0.05,              # Steepness threshold
    min_cluster_size=5    # Minimum cluster size
)

labels = optics.fit_predict(X_scaled)
reachability = optics.reachability_[optics.ordering_]
```

## HDBSCAN - Hierarchical DBSCAN

**Advantages over DBSCAN:**
- Automatically finds optimal eps parameter
- Handles varying density clusters better
- Produces hierarchical clustering
- Probabilistic cluster assignments
- More robust to noise

```python
# Install: pip install hdbscan
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=None,
    metric='euclidean'
)

labels = clusterer.fit_predict(X_scaled)

# Soft assignments (probabilities)
soft_clusters = hdbscan.approximate_predict(clusterer, X_scaled)
```

## Comparison of Clustering Methods

| Property | K-Means | Hierarchical | DBSCAN |
|----------|---------|-------------|--------|
| Cluster shape | Spherical | Any | Any |
| Specify k | Yes | No (dendrogram) | No |
| Outlier detection | No | No | Yes |
| Scalability | Excellent | Poor | Good (with indexing) |
| Varying density | Poor | Fair | Fair (standard DBSCAN) |
| Parameter tuning | Moderate | Easy | Difficult |
| Deterministic | No | Yes | Yes |

## When to Use Each Method

**Use K-Means when:**
- You know the number of clusters
- Clusters are roughly spherical
- Computational speed is critical
- Large-scale datasets

**Use Hierarchical when:**
- You need interpretability (dendrogram)
- Want to explore different cluster levels
- Dataset is medium-sized (<10k points)
- Need deterministic results

**Use DBSCAN when:**
- Cluster shape is irregular
- Want automatic outlier detection
- Clusters have similar densities
- Don't know number of clusters

**Use HDBSCAN when:**
- Clusters have varying densities
- Need automatic parameter selection
- Want probabilistic assignments
- Can afford slight computational overhead

## Practical Tips

1. **Always standardize data** - Distance metrics are scale-dependent
2. **Use k-distance graph** - Visual aid for eps selection
3. **Visualize results** - Check if clusters make sense
4. **Try multiple metrics** - Euclidean, Manhattan, Cosine
5. **Validate with multiple indices** - Silhouette, Davies-Bouldin
6. **Combine methods** - Use DBSCAN for exploration, K-Means for production
7. **Handle noise carefully** - Don't ignore outliers, investigate them

## Common Pitfalls

- **Wrong eps** - Using k-distance graph incorrectly
- **Ignoring data distribution** - Not checking density of clusters
- **High dimensionality curse** - Distance becomes meaningless in high-D
- **Not validating** - Assuming algorithm found "truth"
- **Single metric fixation** - Not trying different distance metrics
- **Treating -1 label as junk** - Noise points may contain valuable insights

## Evaluation Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score (only for valid clusters, not noise)
silhouette = silhouette_score(X_scaled, labels[labels != -1])

# Davies-Bouldin Index (lower is better)
davies_bouldin = davies_bouldin_score(X_scaled, labels)

# Calinski-Harabasz Index (higher is better)
from sklearn.metrics import calinski_harabasz_score
calinski = calinski_harabasz_score(X_scaled, labels)
```

## Resources for Advanced Study

- Original DBSCAN paper: "A Density-Based Algorithm for Discovering Clusters"
- OPTICS paper: "OPTICS: Ordering Points To Identify Clustering Structure"
- HDBSCAN documentation and research papers
- Scikit-learn clustering user guide
