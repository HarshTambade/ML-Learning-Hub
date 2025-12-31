"""Binary Classification using Logistic Regression

This script demonstrates binary classification using logistic regression
with comprehensive examples, evaluation metrics, and visualizations.

Author: ML Learning Hub
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    auc, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. CREATE SYNTHETIC DATASET
# ============================================================================

def create_binary_classification_dataset(n_samples=1000, noise=0.2):
    """
    Create a synthetic binary classification dataset.
    """
    X = np.random.randn(n_samples, 2) * 3
    y = (2 * X[:, 0] + X[:, 1] + np.random.randn(n_samples) * noise) > 0
    y = y.astype(int)
    return X, y

X, y = create_binary_classification_dataset(n_samples=1000)
print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
print()

# ============================================================================
# 2. SPLIT DATA AND SCALE FEATURES
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split and scaling completed")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print()

# ============================================================================
# 3. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================

model = LogisticRegression(
    penalty='l2',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model Training Summary:")
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]}")
print()

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Predictions generated")
print()

# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

def evaluate_model(y_true, y_pred, y_proba=None, set_name=""):
    """
    Calculate comprehensive evaluation metrics.
    """
    print(f"\n{set_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    
    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model(y_train, y_train_pred, set_name="Training Set")
evaluate_model(y_test, y_test_pred, y_test_pred_proba, set_name="Test Set")

# ============================================================================
# 6. CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation Scores (5-fold): {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print()

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Logistic Regression: Binary Classification', fontsize=16, fontweight='bold')

# Decision Boundary
ax1 = axes[0, 0]
h = 0.02
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.4)
ax1.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', label='Class 0')
ax1.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', label='Class 1')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Decision Boundary')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_title('Confusion Matrix')

# ROC Curve
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Probability Distribution
ax4 = axes[1, 1]
ax4.hist(y_test_pred_proba[y_test==0], bins=20, alpha=0.5, label='Class 0', color='blue')
ax4.hist(y_test_pred_proba[y_test==1], bins=20, alpha=0.5, label='Class 1', color='red')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Frequency')
ax4.set_title('Prediction Probability Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 9. SIGMOID FUNCTION VISUALIZATION
# ============================================================================

z = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 5))
plt.plot(z, sigmoid, 'b-', linewidth=2, label='Sigmoid Function')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function: σ(z) = 1 / (1 + e^(-z))')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\n" + "="*70)
print("Binary Classification Training Complete!")
print("="*70)
