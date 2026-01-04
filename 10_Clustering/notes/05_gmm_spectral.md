# 05: Gaussian Mixture Models and Spectral Clustering

## Part 1: Gaussian Mixture Models (GMM)

### Overview
Gaussian Mixture Models represent data as arising from a mixture of Gaussian distributions. Unlike K-Means (hard assignment), GMM provides probabilistic soft assignments - each point has a probability of belonging to each cluster.

### Mathematical Foundation

**Mixture Model:**
```
P(x) = Σ_{k=1}^{K} π_k * N(x | μ_k, Σ_k)
```
Where:
- π_k: Mixing coefficient (prior probability of cluster k)
- N(x | μ_k, Σ_k): Gaussian distribution with mean μ_k and covariance Σ_k
- K: Number of Gaussian components

**Gaussian Distribution:**
```
N(x | μ, Σ) = (1 / sqrt((2π)^D |Σ|)) * exp(-0.5 * (x-μ)^T Σ^{-1} (x-μ))
```

### EM Algorithm (Expectation-Maximization)

**E-Step (Expectation):**
Compute responsibility (soft assignment) of each Gaussian for each point:
```
γ_nk = (π_k * N(x_n | μ_k, Σ_k)) / Σ_j π_j * N(x_n | μ_j, Σ_j)
```

**M-Step (Maximization):**
Update parameters using expected assignments:

**Update mixing coefficients:**
```
π_k = (1/N) * Σ_n γ_nk
```

**Update means:**
```
μ_k = (Σ_n γ_nk * x_n) / Σ_n γ_nk
```

**Update covariances:**
```
Σ_k = (Σ_n γ_nk * (x_n - μ_k)(x_n - μ_k)^T) / Σ_n γ_nk
```

### Likelihood Function

EM algorithm maximizes log-likelihood:
```
log p(X | π, μ, Σ) = Σ_n log(Σ_k π_k * N(x_n | μ_k, Σ_k))
```

### Covariance Matrix Options

| Type | Description | Parameters | Speed |
|------|-------------|-----------|-------|
| Full | Full covariance | d(d+1)/2 per cluster | Slow |
| Tied | Shared covariance | d(d+1)/2 total | Faster |
| Diag | Diagonal covariance | d per cluster | Much faster |
| Spherical | Spherical covariance | 1 per cluster | Fastest |

### Model Selection Criteria

**Bayesian Information Criterion (BIC):**
```
BIC = -2 * log(L) + p * log(N)
```
Where L = likelihood, p = parameters, N = samples
Lower is better.

**Akaike Information Criterion (AIC):**
```
AIC = -2 * log(L) + 2 * p
```
More lenient than BIC.

### Scikit-learn Implementation

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Fit GMM
gmm = GaussianMixture(
    n_components=3,              # Number of Gaussians
    covariance_type='full',      # Options: full, tied, diag, spherical
    max_iter=100,
    random_state=42,
    n_init=10                    # Number of initializations
)

# Fit model
gmm.fit(X_scaled)

# Hard assignments
labels = gmm.predict(X_scaled)

# Soft assignments (probabilities)
soft_assignments = gmm.predict_proba(X_scaled)

# Log-likelihood
log_likelihood = gmm.score(X_scaled)

# BIC and AIC
bic = gmm.bic(X_scaled)
aic = gmm.aic(X_scaled)

# Access parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_  # π_k
```

### Finding Optimal Number of Clusters

```python
import numpy as np
import matplotlib.pyplot as plt

bic_scores = []
aic_scores = []
K_range = range(1, 11)

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, bic_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.title('BIC vs K')

plt.subplot(1, 2, 2)
plt.plot(K_range, aic_scores, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC')
plt.title('AIC vs K')
plt.tight_layout()
plt.show()
```

### Advantages of GMM

1. **Probabilistic framework** - Provides confidence scores
2. **Soft assignments** - Points can belong to multiple clusters
3. **Principled model selection** - BIC, AIC provide objective criteria
4. **Interpretable** - Each cluster has mean and covariance
5. **Flexible covariances** - Can model elliptical clusters
6. **Uncertainty quantification** - Measures of confidence in assignments

### Limitations of GMM

1. **Computationally expensive** - EM iteration with covariance updates
2. **Local optima** - EM can converge to suboptimal solutions
3. **Singularity issues** - Covariance matrix can become singular
4. **Assumes Gaussian distribution** - May not fit non-Gaussian data
5. **Sensitive to initialization** - Need multiple random starts
6. **Requires choosing k** - Still need to specify number of clusters

---

## Part 2: Spectral Clustering

### Overview
Spectral clustering uses eigenvalues (spectrum) of similarity/distance matrices to perform dimensionality reduction before clustering. Effective for non-convex clusters and complex geometries.

### Key Concepts

**Similarity Graph:**
Represent data as weighted graph where edge weights represent similarity between points.

**Graph Laplacian:**
Captures structure of similarity graph.

### Graph Laplacian Matrix

**Unnormalized Laplacian:**
```
L = D - W
```
Where:
- W: Adjacency matrix (similarity/affinity matrix)
- D: Degree matrix (diagonal matrix with D_ii = Σ_j W_ij)

**Normalized Laplacian (Symmetric):**
```
L_norm = I - D^{-1/2} W D^{-1/2}
```

**Normalized Laplacian (Random Walk):**
```
L_rw = I - D^{-1} W
```

### Similarity Graphs

**1. Epsilon-Neighborhood Graph:**
```
W_ij = 1 if ||x_i - x_j|| < ε
W_ij = 0 otherwise
```
Simple but sensitive to ε choice.

**2. k-Nearest Neighbors Graph:**
```
W_ij = 1 if x_j is among k nearest neighbors of x_i
```
More stable, automatic parameter (k).

**3. Gaussian Similarity:**
```
W_ij = exp(-||x_i - x_j||² / (2σ²))
```
Smooth, differentiable weights.
Sensitive to σ parameter.

### Spectral Clustering Algorithm

1. Construct similarity/affinity matrix W
2. Compute degree matrix D
3. Compute graph Laplacian L (or normalized version)
4. Compute eigenvectors corresponding to K smallest eigenvalues
5. Stack eigenvectors as columns to form matrix U ∈ ℝ^{n×K}
6. Treat rows of U as points and cluster using K-Means

### Mathematical Intuition

Eigenvectors of Laplacian capture cluster structure:
- Points in same cluster have similar eigenvector values
- Spectral gap (difference between eigenvalues) indicates number of clusters

### Scikit-learn Implementation

```python
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Spectral clustering with precomputed affinity
spectral = SpectralClustering(
    n_clusters=3,              # Number of clusters
    affinity='nearest_neighbors',  # How to compute affinity
    n_neighbors=10,            # For k-nearest neighbors
    assign_labels='kmeans',    # How to label eigenvectors
    random_state=42
)

labels = spectral.fit_predict(X_scaled)

# Alternative: using Gaussian affinity
spectral_gaussian = SpectralClustering(
    n_clusters=3,
    affinity='rbf',            # Gaussian/RBF kernel
    gamma=1.0,                 # σ parameter (1/(2σ²))
    assign_labels='kmeans'
)

labels = spectral_gaussian.fit_predict(X_scaled)
```

### Determining Optimal Number of Clusters

**Spectral Gap Method:**
```python
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel

# Compute similarity matrix
W = rbf_kernel(X_scaled, gamma=gamma)

# Degree matrix
D = np.diag(W.sum(axis=1))

# Laplacian
L = D - W

# Compute eigenvalues (smallest K)
eigenvalues, eigenvectors = eigsh(L, k=10, which='SM')

# Look for largest gap in eigenvalues
gaps = np.diff(eigenvalues)
optimal_k = np.argmax(gaps) + 1

plt.plot(eigenvalues)
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalue Spectrum')
plt.show()
```

### Advantages of Spectral Clustering

1. **Handles arbitrary cluster shapes** - Non-convex, non-spherical clusters
2. **Graph-based** - Captures data manifold structure
3. **Theoretically grounded** - Spectral graph theory foundation
4. **Works with similarity matrices** - Flexible distance/similarity definitions
5. **Can handle high-dimensional data** - Via dimensionality reduction
6. **Interpretable eigenvalues** - Spectral gap indicates cluster structure

### Limitations of Spectral Clustering

1. **Parameter sensitive** - Choice of similarity function and parameters crucial
2. **Computational cost** - Eigenvalue decomposition O(n³) for full graphs
3. **Memory intensive** - Stores n×n similarity matrix
4. **Parameter selection difficult** - No clear guidance for σ, k, ε
5. **Eigenvector interpretation** - K-Means on eigenvectors may introduce artifacts
6. **Not invariant to scaling** - Still need to standardize features

### Comparison of GMM and Spectral Clustering

| Aspect | GMM | Spectral |
|--------|-----|----------|
| Cluster shape | Elliptical (Gaussian) | Arbitrary |
| Soft assignments | Yes | No (hard) |
| Parameter selection | BIC/AIC | Spectral gap |
| Computational complexity | O(n*K*D*T) | O(n³) |
| Scalability | Better | Poorer |
| Similarity matrix | No (uses data) | Uses W matrix |
| Theoretical foundation | Probabilistic | Graph-based |

### When to Use GMM

- Data appears Gaussian distributed
- Need probabilistic assignments
- Computational efficiency important
- Want principled model selection (BIC/AIC)
- Can afford EM iterations

### When to Use Spectral Clustering

- Non-convex, complex geometries
- Graph-structured data
- Need to incorporate custom similarity
- Small-to-medium datasets
- Interpretability via graph structure important

## Advanced Applications

### Combining GMM and Spectral
```python
# Use spectral for initial dimensionality reduction
from sklearn.decomposition import SpectralEmbedding

# Reduce dimensionality
se = SpectralEmbedding(n_components=3)
X_embedded = se.fit_transform(X_scaled)

# Apply GMM on embedded space
gmm = GaussianMixture(n_components=3).fit(X_embedded)
labels = gmm.predict(X_embedded)
```

## Practical Tips

1. **Always standardize** - Scale features before using distance metrics
2. **Use multiple metrics** - Try different similarity functions
3. **Validate thoroughly** - Silhouette scores, Davies-Bouldin, Calinski-Harabasz
4. **Visualize eigenvectors** - Look for distinct patterns in first K eigenvectors
5. **Multiple initializations** - GMM especially sensitive to initialization
6. **Check spectral gap** - First indicator of optimal number of clusters

## Resources

- "A Tutorial on Spectral Clustering" - Von Luxburg, 2007
- GMM theory: Bishop's "Pattern Recognition and Machine Learning"
- Scikit-learn clustering documentation
- Graph-based clustering literature
