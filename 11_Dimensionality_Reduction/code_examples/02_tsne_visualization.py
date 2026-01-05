"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) Visualization

Demonstrates:
1. 2D and 3D t-SNE visualizations
2. Perplexity parameter tuning
3. Learning rate and iteration effects
4. Comparison with PCA
5. Computational time analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
import time

# Load datasets
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print("="*70)
print("t-SNE Visualization Analysis")
print("="*70)

# Standardize data
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_digits_scaled = scaler.fit_transform(X_digits)

# ================================================================
# 1. Basic 2D t-SNE Visualization
# ================================================================
print("\n1. Computing 2D t-SNE (perplexity=30)...")
start_time = time.time()
tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_iris_scaled)
tsne_time_2d = time.time() - start_time
print(f"   Completed in {tsne_time_2d:.3f} seconds")

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i in range(len(np.unique(y_iris))):
    indices = y_iris == i
    ax.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1],
              c=colors[i], label=iris.target_names[i],
              s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE 2D Visualization of Iris Dataset', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 2. 3D t-SNE Visualization
# ================================================================
print("\n2. Computing 3D t-SNE (perplexity=30)...")
start_time = time.time()
tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_iris_scaled)
tsne_time_3d = time.time() - start_time
print(f"   Completed in {tsne_time_3d:.3f} seconds")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(np.unique(y_iris))):
    indices = y_iris == i
    ax.scatter(X_tsne_3d[indices, 0], X_tsne_3d[indices, 1], X_tsne_3d[indices, 2],
              c=colors[i], label=iris.target_names[i],
              s=100, alpha=0.7, edgecolors='black')
ax.set_xlabel('t-SNE 1', fontsize=10)
ax.set_ylabel('t-SNE 2', fontsize=10)
ax.set_zlabel('t-SNE 3', fontsize=10)
ax.set_title('t-SNE 3D Visualization of Iris Dataset', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ================================================================
# 3. Perplexity Parameter Study
# ================================================================
print("\n3. Studying Perplexity Effects...")
perplexities = [5, 10, 20, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, perp in enumerate(perplexities):
    print(f"   Computing t-SNE with perplexity={perp}...")
    tsne_temp = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42)
    X_temp = tsne_temp.fit_transform(X_iris_scaled)
    
    ax = axes[idx]
    for i in range(len(np.unique(y_iris))):
        indices = y_iris == i
        ax.scatter(X_temp[indices, 0], X_temp[indices, 1],
                  c=colors[i], label=iris.target_names[i],
                  s=80, alpha=0.7, edgecolors='black')
    ax.set_title(f'Perplexity = {perp}', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)

axes[-1].axis('off')
plt.tight_layout()
plt.show()

# ================================================================
# 4. t-SNE vs PCA Comparison
# ================================================================
print("\n4. Comparing t-SNE vs PCA...")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_iris_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for i in range(len(np.unique(y_iris))):
    indices = y_iris == i
    ax1.scatter(X_pca_2d[indices, 0], X_pca_2d[indices, 1],
               c=colors[i], label=iris.target_names[i],
               s=100, alpha=0.7, edgecolors='black')
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
ax1.set_title('PCA Visualization', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# t-SNE
for i in range(len(np.unique(y_iris))):
    indices = y_iris == i
    ax2.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1],
               c=colors[i], label=iris.target_names[i],
               s=100, alpha=0.7, edgecolors='black')
ax2.set_xlabel('t-SNE 1', fontsize=12)
ax2.set_ylabel('t-SNE 2', fontsize=12)
ax2.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================================
# 5. Learning Rate and Iterations Study
# ================================================================
print("\n5. Studying Learning Rate Effects...")
learning_rates = [10, 50, 100, 200]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for idx, lr in enumerate(learning_rates):
    print(f"   Computing t-SNE with learning_rate={lr}...")
    tsne_temp = TSNE(n_components=2, learning_rate=lr, n_iter=500, random_state=42)
    X_temp = tsne_temp.fit_transform(X_iris_scaled)
    
    ax = axes[idx]
    for i in range(len(np.unique(y_iris))):
        indices = y_iris == i
        ax.scatter(X_temp[indices, 0], X_temp[indices, 1],
                  c=colors[i], label=iris.target_names[i],
                  s=80, alpha=0.7, edgecolors='black')
    ax.set_title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================================
# 6. Digits Dataset (Larger Dataset)
# ================================================================
print("\n6. t-SNE on Digits Dataset (1797 samples, 64 features)...")
start_time = time.time()
tsne_digits = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_digits_tsne = tsne_digits.fit_transform(X_digits_scaled)
digits_time = time.time() - start_time
print(f"   Completed in {digits_time:.3f} seconds")

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1],
                    c=y_digits, cmap='tab10', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE Visualization of Handwritten Digits Dataset', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Digit (0-9)', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 7. Computational Time Analysis
# ================================================================
print("\n" + "="*70)
print("Computational Time Summary")
print("="*70)
print(f"\nIris Dataset (150 samples, 4 features):")
print(f"  PCA 2D: < 0.01 seconds")
print(f"  t-SNE 2D: {tsne_time_2d:.3f} seconds")
print(f"  t-SNE 3D: {tsne_time_3d:.3f} seconds")
print(f"\nDigits Dataset (1797 samples, 64 features):")
print(f"  t-SNE 2D: {digits_time:.3f} seconds")

print(f"\nNote: t-SNE is significantly slower than PCA but provides")
print(f"      better visualization of local structure.")
print("\n" + "="*70)
print("t-SNE Analysis Complete!")
print("="*70)
