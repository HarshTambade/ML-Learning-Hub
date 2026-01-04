"""Hierarchical Clustering - Agglomerative and Dendrogram Visualization

Demonstrates:
1. Agglomerative clustering
2. Different linkage methods
3. Dendrogram visualization
4. Dendrog cutting for cluster formation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Generate data
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Test different linkage methods
linkage_methods = ['complete', 'single', 'average', 'ward']

for idx, method in enumerate(linkage_methods):
    # Compute linkage
    Z = linkage(X_scaled, method=method)
    
    # Dendrogram in first row
    ax = axes[0, idx if idx < 3 else 2]
    dendrogram(Z, ax=ax, leaf_rotation=90)
    ax.set_title(f'Dendrogram - {method.capitalize()} Linkage')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    
    # Clustering
    clusterer = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = clusterer.fit_predict(X_scaled)
    
    # Plot clusters in second row
    ax = axes[1, idx if idx < 3 else 2]
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'Clusters - {method.capitalize()}')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# Test optimal number of clusters
print("\nSilhouette Scores for different cluster numbers:")
for n_clusters in range(2, 8):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"n_clusters={n_clusters}: {score:.3f}")
