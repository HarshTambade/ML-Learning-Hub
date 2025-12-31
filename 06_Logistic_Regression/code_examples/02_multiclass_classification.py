"""Multiclass Classification using Logistic Regression

This script demonstrates multiclass classification with logistic regression
using the One-vs-Rest (OvR) strategy and different solvers.

Author: ML Learning Hub
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, heatmap
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. LOAD IRIS DATASET
# ============================================================================

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset: Iris")
print(f"Features: {feature_names}")
print(f"Target Classes: {target_names}")
print(f"Shape: X={X.shape}, y={y.shape}")
print(f"Class Distribution: {np.bincount(y)}")
print()

# ============================================================================
# 2. TRAIN-TEST SPLIT AND SCALING
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing completed")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print()

# ============================================================================
# 3. MULTICLASS LOGISTIC REGRESSION
# ============================================================================

# Using multinomial logistic regression
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Logistic Regression Model Trained")
print(f"Classes: {model.classes_}")
print(f"Coefficients shape: {model.coef_.shape}")
print()

# ============================================================================
# 4. PREDICTIONS
# ============================================================================

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

print("Predictions generated")
print()

# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

def evaluate_multiclass(y_true, y_pred, set_name=""):
    print(f"\n{set_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\nPer-Class Scores:")
    for i, class_name in enumerate(target_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"  {class_name}: Accuracy={class_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

evaluate_multiclass(y_train, y_train_pred, "Training Set")
evaluate_multiclass(y_test, y_test_pred, "Test Set")

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
# 8. COMPARISON OF DIFFERENT SOLVERS
# ============================================================================

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag']
scores = []

print("\nSolver Comparison:")
for solver in solvers:
    try:
        model_solver = LogisticRegression(
            multi_class='multinomial',
            solver=solver,
            max_iter=1000,
            random_state=42
        )
        model_solver.fit(X_train_scaled, y_train)
        score = model_solver.score(X_test_scaled, y_test)
        scores.append(score)
        print(f"{solver:15s}: Accuracy = {score:.4f}")
    except Exception as e:
        print(f"{solver:15s}: Not applicable for multinomial")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multiclass Logistic Regression: Iris Dataset', fontsize=16, fontweight='bold')

# Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=target_names, yticklabels=target_names)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_title('Confusion Matrix')

# Prediction Probability for Each Class
ax2 = axes[0, 1]
for i, class_name in enumerate(target_names):
    ax2.hist(y_test_proba[:, i], alpha=0.5, label=class_name, bins=10)
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Probability Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Feature Importance (Coefficients)
ax3 = axes[1, 0]
coef_data = pd.DataFrame(
    model.coef_.T,
    columns=target_names,
    index=feature_names
)
coef_data.plot(kind='barh', ax=ax3)
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Feature Importance (Model Coefficients)')
ax3.legend(title='Class', loc='best')
ax3.grid(True, alpha=0.3, axis='x')

# Accuracy per class
ax4 = axes[1, 1]
class_accuracies = []
for i, class_name in enumerate(target_names):
    class_mask = y_test == i
    acc = accuracy_score(y_test[class_mask], y_test_pred[class_mask])
    class_accuracies.append(acc)
ax4.bar(target_names, class_accuracies, color=['blue', 'green', 'red'])
ax4.set_ylabel('Accuracy')
ax4.set_title('Per-Class Accuracy')
ax4.set_ylim([0, 1.1])
ax4.axhline(y=accuracy_score(y_test, y_test_pred), color='k', linestyle='--', label='Overall')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Multiclass Classification Training Complete!")
print("="*70)
