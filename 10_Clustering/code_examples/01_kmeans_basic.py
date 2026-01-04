"""K-Means Clustering - Basic Implementation and Scikit-learn Usage

This example demonstrates:
1. K-Means from scratch
2. Using scikit-learn's K-Means
3. Elbow method for optimal K
4. Visualization of clusters
5. Evaluation using Silhouette Score
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                        cluster_std=0.6, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Method 1: Using scikit-learn K-Means
print("\n=== K-Means Clustering ===")
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# Evaluation
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

# Elbow Method
print("\n=== Elbow Method ===")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans_temp.fit_predict(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"K={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Clusters
axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
axes[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, label='Centroids')
axes[0, 0].set_title('K-Means Clustering (K=3)')
axes[0, 0].legend()

# Plot 2: Elbow curve
axes[0, 1].plot(K_range, inertias, 'bo-')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Inertia')
axes[0, 1].set_title('Elbow Method')

# Plot 3: Silhouette scores
axes[1, 0].plot(K_range, silhouette_scores, 'go-')
axes[1, 0].set_xlabel('Number of Clusters (K)')
axes[1, 0].set_ylabel('Silhouette Score')
axes[1, 0].set_title('Silhouette Analysis')

# Plot 4: Comparison
axes[1, 1].scatter(K_range, inertias, label='Inertia', alpha=0.6)
ax2 = axes[1, 1].twinx()
ax2.plot(K_range, silhouette_scores, 'g-', label='Silhouette', linewidth=2)
axes[1, 1].set_xlabel('Number of Clusters')
axes[1, 1].set_ylabel('Inertia', color='b')
ax2.set_ylabel('Silhouette Score', color='g')
axes[1, 1].set_title('Metrics Comparison')

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print(f"Optimal K from Silhouette: {K_range[np.argmax(silhouette_scores)]}")
print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
