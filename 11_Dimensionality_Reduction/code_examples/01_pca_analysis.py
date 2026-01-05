"""
Principal Component Analysis (PCA) - Comprehensive Implementation

Demonstrates:
1. Variance analysis and explained variance ratio
2. Scree plot and cumulative explained variance
3. Data reconstruction from reduced dimensions
4. Comparison of different n_components
5. Visualization of principal components
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_classification
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print("=" * 60)
print("PCA Analysis on Iris Dataset")
print(f"Original dimensions: {X.shape}")
print("=" * 60)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 1. Explained Variance Analysis
# ============================================================
pca = PCA()
pca.fit(X_scaled)

print("\nExplained Variance Ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

cumsum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative Explained Variance:")
for i, cum_var in enumerate(cumsum_var):
    print(f"  PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")

print(f"\nVariance by component (eigenvalues):")
for i, var in enumerate(pca.explained_variance_):
    print(f"  PC{i+1}: {var:.4f}")

# ============================================================
# 2. Scree Plot and Cumulative Variance
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.bar(range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
ax1.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_, 'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Cumulative variance
ax2.plot(range(1, len(cumsum_var)+1), cumsum_var,
         'go-', linewidth=2, markersize=8, label='Cumulative')
ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% variance')
ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% variance')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.05])

plt.tight_layout()
plt.show()

# ============================================================
# 3. 2D and 3D Visualization
# ============================================================
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# 2D plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i in range(len(np.unique(y))):
    indices = y == i
    ax.scatter(X_pca_2d[indices, 0], X_pca_2d[indices, 1],
              c=colors[i], label=iris.target_names[i],
              s=100, alpha=0.7, edgecolors='black')

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
ax.set_title('PCA: 2D Projection of Iris Dataset', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(np.unique(y))):
    indices = y == i
    ax.scatter(X_pca_3d[indices, 0], X_pca_3d[indices, 1], X_pca_3d[indices, 2],
              c=colors[i], label=iris.target_names[i],
              s=100, alpha=0.7, edgecolors='black')

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=10)
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)', fontsize=10)
ax.set_title('PCA: 3D Projection of Iris Dataset', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ============================================================
# 4. Data Reconstruction
# ============================================================
print("\n" + "="*60)
print("Data Reconstruction Analysis")
print("="*60)

# Reconstruct with different numbers of components
reconstruction_errors = []
components_range = range(1, len(X_scaled[0])+1)

for n_comp in components_range:
    pca_temp = PCA(n_components=n_comp)
    X_reduced = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_reduced)
    
    # Mean Squared Reconstruction Error
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    reconstruction_errors.append(mse)
    print(f"  n_components={n_comp}: MSE = {mse:.6f}")

# Plot reconstruction error
plt.figure(figsize=(10, 6))
plt.plot(components_range, reconstruction_errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
plt.title('Reconstruction Error vs Number of Components', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 5. Principal Components (Loading Vectors)
# ============================================================
print("\n" + "="*60)
print("Principal Component Loadings")
print("="*60)

pca_full = PCA()
pca_full.fit(X_scaled)

print("\nPC1 Loadings:")
for i, feature in enumerate(feature_names):
    print(f"  {feature}: {pca_full.components_[0, i]:.4f}")

print("\nPC2 Loadings:")
for i, feature in enumerate(feature_names):
    print(f"  {feature}: {pca_full.components_[1, i]:.4f}")

# Biplot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the data points
for i in range(len(np.unique(y))):
    indices = y == i
    ax.scatter(X_pca_2d[indices, 0], X_pca_2d[indices, 1],
              c=colors[i], label=iris.target_names[i],
              s=100, alpha=0.7, edgecolors='black')

# Plot loading vectors
loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
for i, feature in enumerate(feature_names):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
            head_width=0.1, head_length=0.1, fc='darkred', ec='darkred', linewidth=2)
    ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature,
           fontsize=11, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
ax.set_title('PCA Biplot - Data Points and Loading Vectors', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("PCA Analysis Complete!")
print("="*60)
