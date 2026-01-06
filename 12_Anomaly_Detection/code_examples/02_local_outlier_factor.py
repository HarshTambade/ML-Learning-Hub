02_local_outlier_factor.py"""Local Outlier Factor - Density-based Anomaly Detection

LOF detects local density deviations by comparing the density of each point
with the density of its neighbors. Anomalies have significantly lower densities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def generate_data_with_outliers(n_samples=300, contamination=0.1, random_state=42):
    np.random.seed(random_state)
    n_normal = int(n_samples * (1 - contamination))
    
    # Normal data in 2 clusters
    X_normal1 = np.random.randn(n_normal // 2, 2) + [2, 2]
    X_normal2 = np.random.randn(n_normal // 2, 2) + [-2, -2]
    X_normal = np.vstack([X_normal1, X_normal2])
    
    # Outliers scattered around
    n_outliers = n_samples - n_normal
    X_outliers = np.random.uniform(-4, 4, (n_outliers, 2))
    
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])
    
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]

def train_lof(X, n_neighbors=20, contamination=0.1):
    """Train LOF model"""
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False
    )
    predictions = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    return lof, predictions, scores

def evaluate_and_visualize(X, y_true, y_pred, scores, lof_model):
    """Evaluate model and create visualizations"""
    y_pred_binary = (y_pred == -1).astype(int)
    
    print("\n=== LOF Results ===")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Anomaly']))
    print(f"ROC-AUC: {roc_auc_score(y_true, -scores):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y_pred_binary, cmap='coolwarm', s=50)
    axes[0].set_title('LOF: Predicted Anomalies')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=axes[0], label='Anomaly')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, -scores)
    auc = roc_auc_score(y_true, -scores)
    axes[1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('lof_results.png', dpi=100)
    plt.show()

def main():
    print("="*50)
    print("Local Outlier Factor - Anomaly Detection")
    print("="*50)
    
    X, y = generate_data_with_outliers(n_samples=300, contamination=0.1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    lof_model, predictions, scores = train_lof(X, n_neighbors=20, contamination=0.1)
    evaluate_and_visualize(X, y, predictions, scores, lof_model)

if __name__ == "__main__":
    main()
