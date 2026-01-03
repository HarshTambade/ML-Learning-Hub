#!/usr/bin/env python3
"""
Distance Metrics Comparison

Compares different distance metrics (Euclidean, Manhattan, Minkowski, Cosine)
and their impact on KNN performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def create_dataset():
    """
    Create a synthetic dataset for testing.
    """
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2,
        n_redundant=0, n_classes=2, random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)

def scale_data(X_train, X_test):
    """
    Scale the features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def compare_metrics():
    """
    Compare different distance metrics.
    """
    X_train, X_test, y_train, y_test = create_dataset()
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']
    results = {}
    
    print("="*50)
    print("DISTANCE METRICS COMPARISON")
    print("="*50 + "\n")
    
    for metric in metrics:
        try:
            knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
            knn.fit(X_train_scaled, y_train)
            
            train_acc = knn.score(X_train_scaled, y_train)
            test_acc = knn.score(X_test_scaled, y_test)
            cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5).mean()
            
            results[metric] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_score': cv_score
            }
            
            print(f"{metric.upper()}:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  CV Score:       {cv_score:.4f}\n")
        except Exception as e:
            print(f"Error with {metric}: {e}\n")
    
    return results, X_train_scaled, X_test_scaled, y_train, y_test

def visualize_results(results):
    """
    Visualize metrics comparison.
    """
    df = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    df[['train_accuracy', 'test_accuracy']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Accuracy Comparison by Distance Metric')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Distance Metric')
    axes[0].legend(['Train', 'Test'])
    axes[0].grid(True, alpha=0.3)
    
    # CV scores
    axes[1].bar(df.index, df['cv_score'], color='steelblue', alpha=0.7)
    axes[1].set_title('Cross-Validation Scores')
    axes[1].set_ylabel('CV Score')
    axes[1].set_xlabel('Distance Metric')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=100, bbox_inches='tight')
    print("Visualization saved as 'metrics_comparison.png'")
    plt.close()

def distance_explanations():
    """
    Explain different distance metrics.
    """
    print("\n" + "="*50)
    print("DISTANCE METRIC EXPLANATIONS")
    print("="*50 + "\n")
    
    print("1. EUCLIDEAN DISTANCE:")
    print("   Formula: d = sqrt(sum((x-y)^2))")
    print("   Use: Most common, good for continuous data\n")
    
    print("2. MANHATTAN DISTANCE:")
    print("   Formula: d = sum(|x-y|)")
    print("   Use: For grid-like data, robust to outliers\n")
    
    print("3. MINKOWSKI DISTANCE:")
    print("   Formula: d = (sum(|x-y|^p))^(1/p)")
    print("   Use: Generalizes Euclidean and Manhattan\n")
    
    print("4. COSINE DISTANCE:")
    print("   Formula: d = 1 - (A·B)/(||A||||B||)")
    print("   Use: For text and high-dimensional sparse data\n")

if __name__ == "__main__":
    print("\n")
    results, X_train, X_test, y_train, y_test = compare_metrics()
    visualize_results(results)
    distance_explanations()
    
    print("\n" + "="*50)
    print("COMPARISON COMPLETED")
    print("="*50 + "\n")
