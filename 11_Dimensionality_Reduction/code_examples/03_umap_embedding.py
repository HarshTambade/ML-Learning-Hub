"""
UMAP (Uniform Manifold Approximation and Projection)
Fast, scalable alternative to t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
import umap
import time

# Load data
iris = load_iris()
X = iris.data
y = iris.target
X_scaled = StandardScaler().fit_transform(X)

print("UMAP Embedding Analysis")
print("=" * 50)

# Basic UMAP
print("\nComputing UMAP...")
start = time.time()
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
print(f"Completed in {time.time()-start:.3f}s")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# UMAP
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = y == i
    axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=colors[i], label=iris.target_names[i],
                   s=100, alpha=0.7, edgecolors='black')
axes[0].set_title('UMAP Embedding', fontsize=14, fontweight='bold')
axes[0].set_xlabel('UMAP 1', fontsize=11)
axes[0].set_ylabel('UMAP 2', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Parameter study
print("\nStudying parameter effects...")
for n_neighbors in [5, 15, 30]:
    model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    embedding = model.fit_transform(X_scaled)
    print(f"  n_neighbors={n_neighbors}: Done")

# Digits dataset
digits = load_digits()
X_digits = StandardScaler().fit_transform(digits.data)
print(f"\nUMAP on Digits (1797 samples)...")
start = time.time()
umap_digits = umap.UMAP(n_components=2, random_state=42)
X_digits_umap = umap_digits.fit_transform(X_digits)
print(f"Completed in {time.time()-start:.3f}s")

scatter = axes[1].scatter(X_digits_umap[:, 0], X_digits_umap[:, 1],
                         c=digits.target, cmap='tab10', s=20, alpha=0.6)
axes[1].set_title('UMAP: Handwritten Digits', fontsize=14, fontweight='bold')
axes[1].set_xlabel('UMAP 1', fontsize=11)
axes[1].set_ylabel('UMAP 2', fontsize=11)
plt.colorbar(scatter, ax=axes[1], label='Digit')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("UMAP Analysis Complete!")
print("="*50)
