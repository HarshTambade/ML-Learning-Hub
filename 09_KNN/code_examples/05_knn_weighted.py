#!/usr/bin/env python3
"""Weighted KNN: Distance-weighted neighbors"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def create_imbalanced_data():
    """Create dataset with class imbalance"""
    X1, y1 = make_blobs(n_samples=100, centers=1, random_state=42, center_box=(-10, 0))
    X2, y2 = make_blobs(n_samples=30, centers=1, random_state=42, center_box=(5, 15))
    X = np.vstack([X1, X2])
    y = np.hstack([y1, np.ones(30, dtype=int)])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def compare_weights():
    """Compare uniform vs distance-weighted KNN"""
    X_train, X_test, y_train, y_test = create_imbalanced_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("="*50)
    print("WEIGHTED KNN COMPARISON")
    print("="*50 + "\n")
    
    # Uniform weights
    knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn_uniform.fit(X_train, y_train)
    acc_uniform = accuracy_score(y_test, knn_uniform.predict(X_test))
    
    # Distance-weighted
    knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_weighted.fit(X_train, y_train)
    acc_weighted = accuracy_score(y_test, knn_weighted.predict(X_test))
    
    print(f"Uniform Weights Accuracy: {acc_uniform:.4f}")
    print(f"Distance-Weighted Accuracy: {acc_weighted:.4f}")
    print(f"Difference: {abs(acc_weighted - acc_uniform):.4f}\n")
    
    # Test across K values
    k_values = range(1, 16)
    uniform_scores = []
    weighted_scores = []
    
    print("Performance across K values:")
    for k in k_values:
        knn_u = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_w = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_u.fit(X_train, y_train)
        knn_w.fit(X_train, y_train)
        uniform_scores.append(knn_u.score(X_test, y_test))
        weighted_scores.append(knn_w.score(X_test, y_test))
        print(f"K={k}: Uniform={uniform_scores[-1]:.4f}, Weighted={weighted_scores[-1]:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, uniform_scores, 'o-', label='Uniform Weights', linewidth=2)
    plt.plot(k_values, weighted_scores, 's-', label='Distance Weighted', linewidth=2)
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Uniform vs Distance-Weighted KNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('weighted_comparison.png', dpi=100, bbox_inches='tight')
    print("\nPlot saved as 'weighted_comparison.png'")
    plt.close()

if __name__ == "__main__":
    compare_weights()
    print("\n" + "="*50)
    print("WEIGHTED KNN ANALYSIS COMPLETED")
    print("="*50)
