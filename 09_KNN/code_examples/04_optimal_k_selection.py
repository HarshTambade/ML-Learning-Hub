#!/usr/bin/env python3
"""Optimal K Selection using cross-validation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def find_optimal_k():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_range = range(1, 31)
    cv_scores = []
    
    print("="*50)
    print("OPTIMAL K SELECTION")
    print("="*50 + "\n")
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"K={k}: CV Score = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    optimal_k = list(k_range)[np.argmax(cv_scores)]
    print(f"\nOptimal K: {optimal_k} with score {max(cv_scores):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cv_scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('K Value')
    plt.ylabel('Cross-Validation Score')
    plt.title('Optimal K Selection')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimal_k.png', dpi=100, bbox_inches='tight')
    print("\nPlot saved as 'optimal_k.png'")
    plt.close()
    
    return optimal_k, cv_scores

if __name__ == "__main__":
    optimal_k, scores = find_optimal_k()
    print("\n" + "="*50)
    print("COMPLETED")
    print("="*50)
