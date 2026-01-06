"""Isolation Forest - Tree-based Anomaly Detection

This implementation demonstrates the Isolation Forest algorithm,
which is effective for detecting anomalies in high-dimensional datasets.

Key Features:
- Tree-based approach
- Efficient for large datasets
- Works well with both point and collective anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns


def generate_anomalous_data(n_samples=1000, n_features=5, contamination=0.05, random_state=42):
    """
    Generate synthetic dataset with anomalies.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    contamination : float
        Proportion of anomalies (0 to 1)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) containing true labels (0=normal, 1=anomaly)
    """
    np.random.seed(random_state)
    
    # Generate normal data
    n_normal = int(n_samples * (1 - contamination))
    normal_data = np.random.randn(n_normal, n_features)
    
    # Generate anomalies (outliers with larger values)
    n_anomalies = n_samples - n_normal
    anomaly_data = np.random.uniform(3, 5, size=(n_anomalies, n_features))
    
    # Combine and shuffle
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


def train_isolation_forest(X_train, contamination=0.05, n_estimators=100, random_state=42):
    """
    Train Isolation Forest model.
    
    Parameters:
    -----------
    X_train : ndarray
        Training data
    contamination : float
        Expected proportion of anomalies
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random seed
    
    Returns:
    --------
    model : IsolationForest
        Trained model
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def predict_anomalies(model, X):
    """
    Predict anomalies using trained model.
    
    Parameters:
    -----------
    model : IsolationForest
        Trained model
    X : ndarray
        Data to predict on
    
    Returns:
    --------
    predictions : ndarray
        Predicted labels (-1=anomaly, 1=normal)
    anomaly_scores : ndarray
        Anomaly scores (lower = more anomalous)
    """
    predictions = model.predict(X)
    anomaly_scores = model.score_samples(X)
    return predictions, anomaly_scores


def evaluate_model(y_true, y_pred, anomaly_scores):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    anomaly_scores : ndarray
        Anomaly scores
    """
    # Convert predictions to binary (0=normal, 1=anomaly)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_binary, 
                              target_names=['Normal', 'Anomaly']))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_true, -anomaly_scores)  # Lower scores = anomalies
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    return cm, roc_auc


def visualize_results(X, y_true, y_pred, anomaly_scores, model):
    """
    Visualize anomaly detection results.
    """
    # Convert predictions to binary
    y_pred_binary = (y_pred == -1).astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scatter plot of first two features
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred_binary, 
                        cmap='coolwarm', alpha=0.6, s=30)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Isolation Forest: Predicted Anomalies\n(First Two Features)')
    plt.colorbar(scatter, ax=ax, label='Anomaly')
    
    # Plot 2: Confusion Matrix
    ax = axes[0, 1]
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    # Plot 3: Anomaly Scores Distribution
    ax = axes[1, 0]
    normal_scores = anomaly_scores[y_true == 0]
    anomaly_scores_true = anomaly_scores[y_true == 1]
    ax.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
    ax.hist(anomaly_scores_true, bins=30, alpha=0.7, label='Anomaly', color='red')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Anomaly Scores')
    ax.legend()
    ax.axvline(x=model.offset_, color='green', linestyle='--', 
              label=f'Decision Boundary: {model.offset_:.3f}')
    
    # Plot 4: ROC Curve
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, -anomaly_scores)
    roc_auc = roc_auc_score(y_true, -anomaly_scores)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
           label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('isolation_forest_results.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved as 'isolation_forest_results.png'")
    plt.show()


def main():
    """
    Main function to demonstrate Isolation Forest.
    """
    print("="*60)
    print("Isolation Forest - Anomaly Detection")
    print("="*60)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic dataset...")
    X, y_true = generate_anomalous_data(n_samples=1000, n_features=5, contamination=0.05)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("[2] Training Isolation Forest model...")
    model = train_isolation_forest(X_scaled, contamination=0.05)
    
    # Predict anomalies
    print("[3] Predicting anomalies...")
    y_pred, anomaly_scores = predict_anomalies(model, X_scaled)
    
    # Evaluate model
    print("[4] Evaluating model performance...")
    cm, roc_auc = evaluate_model(y_true, y_pred, anomaly_scores)
    
    # Visualize results
    print("[5] Creating visualizations...")
    visualize_results(X_scaled, y_true, y_pred, anomaly_scores, model)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(X)}")
    print(f"Actual anomalies: {y_true.sum()}")
    predicted_anomalies = (y_pred == -1).sum()
    print(f"Predicted anomalies: {predicted_anomalies}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
