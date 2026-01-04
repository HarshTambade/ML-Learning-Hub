"""Gaussian Mixture Models - Probabilistic Clustering

Demonstrates:
1. Gaussian Mixture Models (GMM)
2. Expectation-Maximization algorithm
3. Soft vs hard clustering
4. BIC/AIC model selection
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate data
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=0.7)
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Test different numbers of components
n_components_range = range(1, 7)
bic_scores = []
ic_scores = []
silhouette_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    
    bic_scores.append(gmm.bic(X_scaled))
    ic_scores.append(gmm.aic(X_scaled))
    
    labels = gmm.predict(X_scaled)
    if n_components > 1:
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot first 3
    if n_components <= 3:
        ax = axes[0, n_components - 1]
        labels = gmm.predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # Draw ellipses for components
        from matplotlib.patches import Ellipse
        for i, mu in enumerate(gmm.means_):
            covariance = gmm.covariances_[i]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            ellipse = Ellipse(mu, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]),
                            angle=angle, facecolor='none', edgecolor='red')
            ax.add_patch(ellipse)
        ax.set_title(f'GMM with {n_components} components')

# Plot metrics
ax = axes[0, 2]
ax.set_title('Silhouette Scores')
ax.plot(range(2, 7), silhouette_scores, 'go-')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Silhouette Score')

# BIC and AIC
ax = axes[1, 0]
ax.plot(n_components_range, bic_scores, 'b-o', label='BIC')
ax.plot(n_components_range, ic_scores, 'r-o', label='AIC')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Information Criterion')
Add 04_gmm_em.py - Gaussian Mixture Models with EM algorithm and BIC/AIC selectionax.legend()

# Soft probabilities
ax = axes[1, 1]
gmm_optimal = GaussianMixture(n_components=3, random_state=42)
probs = gmm_optimal.fit_predict_proba(X_scaled)
max_prob = np.max(probs, axis=1)
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=max_prob, cmap='RdYlGn')
"""Spectral Clustering - Graph-Based Clustering

Demonstrates:
1. Spectral clustering algorithm
2. Affinity matrix computation
3. Laplacian matrix
4. Handling non-convex clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

# Generate non-convex data
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Test on moon dataset
X_moon, _ = make_moons(n_samples=300, noise=0.05)
X_moon_scaled = StandardScaler().fit_transform(X_moon)

# Test on circle dataset
X_circle, _ = make_circles(n_samples=300, noise=0.05, factor=0.5)
X_circle_scaled = StandardScaler().fit_transform(X_circle)

# Spectral clustering with different n_neighbors
n_neighbors_values = [5, 10, 15]

for idx, n_neighbors in enumerate(n_neighbors_values):
    # Moon dataset
    ax = axes[0, idx]
    spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                             n_neighbors=n_neighbors, random_state=42)
    labels_moon = spec.fit_predict(X_moon_scaled)
    ax.scatter(X_moon_scaled[:, 0], X_moon_scaled[:, 1], c=labels_moon, cmap='viridis')
    ax.set_title(f'Moon: n_neighbors={n_neighbors}')
    
    # Circle dataset
    ax = axes[1, idx]
    spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                             n_neighbors=n_neighbors, random_state=42)
    labels_circle = spec.fit_predict(X_circle_scaled)
    ax.scatter(X_circle_scaled[:, 0], X_circle_scaled[:, 1], c=labels_circle, cmap='viridis')
    ax.set_title(f'Circle: n_neighbors={n_neighbors}')

plt.tight_layout()
plt.show()

print("Spectral Clustering - Moon Dataset:")
for n_neighbors in [3, 5, 10, 15, 20]:
    spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                             n_neighbors=n_neighbors, random_state=42)
    labels = spec.fit_predict(X_moon_scaled)
    score = silhouette_score(X_moon_scaled, labels)
    print(f"n_neighbors={n_neighbors}: Silhouette={score:.3f}")
plt.colorbar(scatter, ax=ax, label='Max Probability')

ax = axes[1, 2]
ax.plot(n_components_range, bic_scores, 'bo-')
ax.axvline(3, color='r', linestyle='--', label='Optimal')
ax.set_xlabel('Number of Components')
ax.set_ylabel('BIC')
ax.set_title('BIC with Optimal K')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Optimal number of components: {np.argmin(bic_scores) + 1}")
