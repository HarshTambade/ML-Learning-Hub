"""Statistical Methods - Classical Anomaly Detection

Include Z-Score, Modified Z-Score, IQR, and Mahalanobis Distance.
These are fast, interpretable, and effective for univariate data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import mahalanobis

def generate_data(n_samples=500, contamination=0.1, random_state=42):
    """Generate 1D data with outliers"""
    np.random.seed(random_state)
    n_normal = int(n_samples * (1 - contamination))
    
    # Normal data
    X_normal = np.random.normal(0, 1, n_normal)
    # Outliers
    X_outliers = np.random.uniform(3, 5, n_samples - n_normal)
    
    X = np.hstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_samples - n_normal)])
    
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]

def zscore_method(X, threshold=3):
    """Z-Score method"""
    z_scores = np.abs(stats.zscore(X))
    return (z_scores > threshold).astype(int)

def modified_zscore_method(X, threshold=3.5):
    """Modified Z-Score using MAD (Median Absolute Deviation)"""
    median = np.median(X)
    mad = np.median(np.abs(X - median))
    modified_z_scores = 0.6745 * (X - median) / mad
    return (np.abs(modified_z_scores) > threshold).astype(int)

def iqr_method(X, multiplier=1.5):
    """Interquartile Range (IQR) method"""
    Q1 = np.percentile(X, 25)
    Q3 = np.percentile(X, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return ((X < lower_bound) | (X > upper_bound)).astype(int)

def mahalanobis_method(X, threshold=3):
    """Mahalanobis Distance method (for multivariate data)"""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    
    if np.linalg.matrix_rank(cov) < X.shape[1]:
        cov += np.eye(cov.shape[0]) * 1e-6
    
    inv_cov = np.linalg.inv(cov)
    distances = [mahalanobis(x, mean, inv_cov) for x in X]
    
    return (np.array(distances) > threshold).astype(int)

def evaluate_method(y_true, y_pred, method_name):
    """Evaluate anomaly detection method"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{method_name}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")

def main():
    print("="*60)
    print("Statistical Methods - Anomaly Detection")
    print("="*60)
    
    X, y_true = generate_data(n_samples=500, contamination=0.1)
    
    # Apply methods
    z_pred = zscore_method(X, threshold=3)
    mod_z_pred = modified_zscore_method(X, threshold=3.5)
    iqr_pred = iqr_method(X, multiplier=1.5)
    maha_pred = mahalanobis_method(X, threshold=3)
    
    # Evaluate
    evaluate_method(y_true, z_pred, "Z-Score")
    evaluate_method(y_true, mod_z_pred, "Modified Z-Score")
    evaluate_method(y_true, iqr_pred, "IQR Method")
    evaluate_method(y_true, maha_pred, "Mahalanobis Distance")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = [z_pred, mod_z_pred, iqr_pred, maha_pred]
    names = ["Z-Score", "Modified Z-Score", "IQR", "Mahalanobis"]
    
    for idx, (pred, name) in enumerate(zip(methods, names)):
        ax = axes[idx // 2, idx % 2]
        normal = X[y_true == 0]
        anomaly = X[y_true == 1]
        pred_anomaly = X[pred == 1]
        
        ax.scatter(range(len(normal)), normal, alpha=0.6, label='Normal', s=20)
        ax.scatter(range(len(normal), len(X)), anomaly, alpha=0.6, label='True Anomaly', s=20)
        ax.scatter(np.where(pred == 1)[0], pred_anomaly, color='red', marker='x', s=100, label='Detected')
        ax.set_title(name)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('statistical_methods_results.png', dpi=100)
    plt.show()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
