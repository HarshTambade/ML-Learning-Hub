#!/usr/bin/env python3
"""
Basic KNN Classification Example

This script demonstrates KNN classification using the Iris dataset.
It shows how to:
- Load and split data
- Scale features
- Train KNN classifier
- Make predictions
- Evaluate performance
- Visualize decision boundaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def load_and_prepare_data():
    """
    Load Iris dataset and prepare for modeling.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("Dataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features (CRITICAL for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, iris


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k_values=[3, 5, 7, 9]):
    """
    Train KNN models with different K values and evaluate.
    """
    results = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn.fit(X_train, y_train)
        
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        results[k] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model': knn,
            'predictions': y_pred_test
        }
        
        print(f"K = {k}:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    return results


def detailed_evaluation(y_test, y_pred, target_names):
    """
    Provide detailed evaluation metrics.
    """
    print("\n" + "="*50)
    print("DETAILED EVALUATION (K=3)")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (K=3)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.close()


def plot_k_comparison(results):
    """
    Visualize accuracy vs K value.
    """
    k_values = list(results.keys())
    train_accs = [results[k]['train_accuracy'] for k in k_values]
    test_accs = [results[k]['test_accuracy'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(k_values, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Accuracy vs K Value', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.ylim([0.9, 1.01])
    plt.tight_layout()
    plt.savefig('k_comparison.png', dpi=100, bbox_inches='tight')
    print("K comparison plot saved as 'k_comparison.png'")
    plt.close()


def demonstrate_prediction(X_train, y_train, iris):
    """
    Show how to make predictions and understand neighbors.
    """
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Create a sample prediction
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa-like flower
    prediction = knn.predict(sample)
    distances, indices = knn.kneighbors(sample)
    
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    print(f"\nSample: {sample[0]}")
    print(f"Predicted Class: {iris.target_names[prediction[0]]}")
    print(f"\n3 Nearest Neighbors:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i+1}. Class: {iris.target_names[y_train[idx]]}, Distance: {dist:.4f}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("KNN CLASSIFICATION EXAMPLE - IRIS DATASET")
    print("="*50 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, iris = load_and_prepare_data()
    
    # Train and evaluate
    print("Training KNN with different K values...\n")
    results = train_and_evaluate_knn(X_train, X_test, y_train, y_test)
    
    # Detailed evaluation with K=3
    best_model = results[3]['model']
    y_pred = best_model.predict(X_test)
    detailed_evaluation(y_test, y_pred, iris.target_names)
    
    # Visualization
    print("\nGenerating visualizations...")
    plot_k_comparison(results)
    
    # Demonstrate prediction
    demonstrate_prediction(X_train, y_train, iris)
    
    print("\n" + "="*50)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")
