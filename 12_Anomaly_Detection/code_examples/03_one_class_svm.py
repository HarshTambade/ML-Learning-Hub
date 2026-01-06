"""One-Class SVM - Kernel-based Anomaly Detection

One-Class SVM learns a decision boundary around normal data points.
It uses kernel methods to handle non-linear decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

def generate_data_with_outliers(n_samples=500, contamination=0.1, random_state=42):
    np.random.seed(random_state)
    n_normal = int(n_samples * (1 - contamination))
    
    # Normal data
    X_normal = np.random.randn(n_normal, 5) * 2
    
    # Outliers
    n_outliers = n_samples - n_normal
    X_outliers = np.random.uniform(-5, 5, (n_outliers, 5))
    
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])
    
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]

def train_one_class_svm(X, nu=0.1, kernel='rbf', gamma='auto'):
    """Train One-Class SVM model"""
    model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)
    return model, predictions, scores

def evaluate_model(y_true, y_pred, scores):
    """Evaluate and visualize results"""
    y_pred_binary = (y_pred == -1).astype(int)
    
    print("\n=== One-Class SVM Results ===")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Anomaly']))
    
    try:
        auc = roc_auc_score(y_true, -scores)
        print(f"ROC-AUC: {auc:.4f}")
        return auc
    except:
        return 0

def main():
    print("="*50)
    print("One-Class SVM - Anomaly Detection")
    print("="*50)
    
    X, y = generate_data_with_outliers(n_samples=500, contamination=0.1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    model, predictions, scores = train_one_class_svm(X, nu=0.1, kernel='rbf')
    evaluate_model(y, predictions, scores)
    
    # Results summary
    y_pred_binary = (predictions == -1).astype(int)
    print(f"\nTotal samples: {len(X)}")
    print(f"Actual anomalies: {y.sum()}")
    print(f"Predicted anomalies: {y_pred_binary.sum()}")
    print("="*50)

if __name__ == "__main__":
    main()
