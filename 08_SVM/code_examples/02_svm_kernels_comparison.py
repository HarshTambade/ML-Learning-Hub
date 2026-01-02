"""
SVM Kernels Comparison
Compares linear, RBF, polynomial, and sigmoid kernels
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate non-linear dataset
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, random_state=42)
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

print("Generated non-linear datasets")
print(f"Circles shape: {X_circles.shape}, Moons shape: {X_moons.shape}\n")

# Scale data
scaler = StandardScaler()
X_circles_scaled = scaler.fit_transform(X_circles)
X_moons_scaled = scaler.fit_transform(X_moons)

# Define kernels to compare
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernelparams = {
    'linear': {'kernel': 'linear'},
    'rbf': {'kernel': 'rbf', 'gamma': 'scale'},
    'poly': {'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
    'sigmoid': {'kernel': 'sigmoid', 'gamma': 'scale'}
}

print("Training SVM models with different kernels...\n")

# Train models and store results
results = {}
for data_name, X, y in [('Circles', X_circles_scaled, y_circles), ('Moons', X_moons_scaled, y_moons)]:
    results[data_name] = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for kernel in kernels:
        svm = SVC(C=1.0, **kernelparams[kernel])
        svm.fit(X_train, y_train)
        
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        n_support = len(svm.support_)
        
        results[data_name][kernel] = {
            'model': svm,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'support_vectors': n_support
        }
        
        print(f"{data_name} - {kernel.upper()}:")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        print(f"  Support vectors: {n_support}\n")

# Visualization function
def plot_decision_boundary(ax, X, y, svm, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=200, linewidth=1.5, facecolors='none', edgecolors='green')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return scatter

# Plot decision boundaries
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('SVM Kernels Comparison - Decision Boundaries', fontsize=14, fontweight='bold')

for row, (data_name, X, y) in enumerate([('Circles', X_circles_scaled, y_circles), ('Moons', X_moons_scaled, y_moons)]):
    for col, kernel in enumerate(kernels):
        ax = axes[row, col]
        svm = results[data_name][kernel]['model']
        plot_decision_boundary(ax, X, y, svm, f"{data_name} - {kernel.upper()}")

plt.tight_layout()
plt.savefig('kernels_comparison.png', dpi=100, bbox_inches='tight')
print("Decision boundary plot saved as 'kernels_comparison.png'")
plt.show()

# Performance comparison bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('SVM Kernels Performance Comparison', fontsize=14, fontweight='bold')

for idx, data_name in enumerate(['Circles', 'Moons']):
    ax = axes[idx]
    
    test_accs = [results[data_name][k]['test_acc'] for k in kernels]
    support_counts = [results[data_name][k]['support_vectors'] for k in kernels]
    
    x = np.arange(len(kernels))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax2.bar(x + width/2, support_counts, width, label='Support Vectors', alpha=0.8, color='salmon')
    
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Test Accuracy', color='skyblue')
    ax2.set_ylabel('Number of Support Vectors', color='salmon')
    ax.set_title(f'{data_name} Dataset', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.set_ylim([0.8, 1.0])
    
    ax.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='salmon')

plt.tight_layout()
plt.savefig('kernels_performance.png', dpi=100, bbox_inches='tight')
print("Performance comparison plot saved as 'kernels_performance.png'\n")
plt.show()

print("="*60)
print("Kernel Comparison Complete!")
print("Key Insights:")
print("- RBF kernel: Best for non-linear data (circles, moons)")
print("- Linear kernel: Best for linearly separable data")
print("- Polynomial kernel: Good for intermediate complexity")
print("- Sigmoid kernel: Similar to neural networks, variable results")
print("="*60)
