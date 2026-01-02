"""
Non-Linear Classification with RBF Kernel
Demonstrates why non-linear kernels are needed
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("Generating non-linear datasets...\n")

# Generate three non-linear datasets
datasets = [
    ('Circles', make_circles(n_samples=300, noise=0.05, random_state=42)),
    ('Moons', make_moons(n_samples=300, noise=0.05, random_state=42)),
    ('Blobs', make_blobs(n_samples=300, centers=3, random_state=42))
]

# Scale all datasets
scaled_datasets = {}
for name, (X, y) in datasets:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_datasets[name] = (X_scaled, y)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Linear vs RBF Kernel for Non-Linear Data', fontsize=16, fontweight='bold')

row = 0
for name, (X, y) in scaled_datasets.items():
    # Original data
    ax = axes[row, 0]
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0', alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1', alpha=0.7)
    ax.set_title(f'{name} Dataset')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Linear kernel
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X, y)
    linear_acc = svm_linear.score(X, y)
    
    ax = axes[row, 1]
    plot_svm_decision_boundary(ax, X, y, svm_linear, f'{name} - Linear (Acc: {linear_acc:.3f})')
    
    # RBF kernel
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X, y)
    rbf_acc = svm_rbf.score(X, y)
    
    ax = axes[row, 2]
    plot_svm_decision_boundary(ax, X, y, svm_rbf, f'{name} - RBF (Acc: {rbf_acc:.3f})')
    
    print(f"{name} Dataset:")
    print(f"  Linear Accuracy: {linear_acc:.4f}")
    print(f"  RBF Accuracy:    {rbf_acc:.4f}")
    print(f"  Improvement:     {(rbf_acc - linear_acc)*100:.2f}%\n")
    
    row += 1

plt.tight_layout()
plt.savefig('non_linear_classification.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved as 'non_linear_classification.png'")
plt.show()

def plot_svm_decision_boundary(ax, X, y, svm, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', edgecolors='k', s=50, alpha=0.7)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', edgecolors='k', s=50, alpha=0.7)
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=200, linewidth=1.5, facecolors='none', edgecolors='green')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

print("="*60)
print("Key Insights:")
print("- Linear kernel cannot separate non-linear data well")
print("- RBF kernel maps data to higher dimension implicitly")
print("- RBF's gamma parameter controls boundary smoothness")
print("  - High gamma: complex boundaries, risk of overfitting")
print("  - Low gamma: smooth boundaries, may underfit")
print("="*60)
