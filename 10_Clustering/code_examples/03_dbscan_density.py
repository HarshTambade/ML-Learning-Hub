"""DBSCAN - Density-Based Spatial Clustering of Applications with Noise

Demonstrates:
1. DBSCAN algorithm
2. Effect of eps and min_samples
3. Handling outliers
4. Comparison with K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate data
X, _ = make_moons(n_samples=300, noise=0.05)
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Test different eps values
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_samples = 5

for idx, eps in enumerate(eps_values):
    ax = axes[0, idx if idx < 3 else 2]
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'eps={eps}, Clusters={n_clusters}, Noise={n_noise}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Test min_samples effect
eps = 0.3
min_samples_values = [3, 5, 10, 15]

for idx, ms in enumerate(min_samples_values):
    ax = axes[1, idx]
    
    dbscan = DBSCAN(eps=eps, min_samples=ms)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'min_samples={ms}, Clusters={n_clusters}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\nOptimal parameters for moon dataset:")
for eps in [0.2, 0.25, 0.3, 0.35]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        print(f"eps={eps}: Silhouette={score:.3f}")
