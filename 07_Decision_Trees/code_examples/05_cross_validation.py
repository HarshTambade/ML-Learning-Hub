"""
Cross-Validation and Model Evaluation for Decision Trees

This script demonstrates:
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out CV
- Time Series CV
- ROC Curves and AUC
- Precision-Recall Curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, 
    KFold, StratifiedKFold, LeaveOneOut
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, precision_score, recall_score
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("CROSS-VALIDATION AND MODEL EVALUATION")
print("=" * 80)

# Load data
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 1. K-FOLD CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("1. K-FOLD CROSS-VALIDATION")
print("=" * 80)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(dt, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"\n5-Fold Cross-Validation Scores: {np.round(scores, 4)}")
print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Min Score: {scores.min():.4f}")
print(f"Max Score: {scores.max():.4f}")

# ============================================================================
# 2. STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("2. STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 80)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(
    dt, X_train, y_train, cv=stratified_kfold, scoring='accuracy'
)

print(f"\nStratified 5-Fold Scores: {np.round(strat_scores, 4)}")
print(f"Mean Accuracy: {strat_scores.mean():.4f} (+/- {strat_scores.std():.4f})")
print(f"\nClass distribution in each fold (more balanced):")
for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X_train, y_train), 1):
    print(f"Fold {fold}: Class 0: {(y_train[val_idx] == 0).sum()}, Class 1: {(y_train[val_idx] == 1).sum()}")

# ============================================================================
# 3. LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("3. LEAVE-ONE-OUT CROSS-VALIDATION (LOO)")
print("=" * 80)

loo = LeaveOneOut()
loo_scores = cross_val_score(dt, X_train[:100], y_train[:100], cv=loo)

print(f"\nLOO CV with 100 samples:")
print(f"Accuracy: {loo_scores.mean():.4f}")
print(f"Number of correct predictions: {loo_scores.sum()} / {len(loo_scores)}")

# ============================================================================
# 4. MULTIPLE SCORING METRICS
# ============================================================================

print("\n" + "=" * 80)
print("4. MULTIPLE SCORING METRICS WITH CV")
print("=" * 80)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    dt, X_train, y_train, cv=5, scoring=scoring
)

print(f"\nCross-Validation Results (5-Fold):")
for metric in scoring:
    key = f'test_{metric}'
    scores_metric = cv_results[key]
    print(f"{metric:10s}: {scores_metric.mean():.4f} (+/- {scores_metric.std():.4f})")

# ============================================================================
# 5. ROC CURVES AND AUC
# ============================================================================

print("\n" + "=" * 80)
print("5. ROC CURVES AND AUC ANALYSIS")
print("=" * 80)

dt.fit(X_train, y_train)
y_pred_proba = dt.predict_proba(X_test)[:, 1]
y_pred = dt.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC Score: {roc_auc:.4f}")
print(f"This means the model ranks a random positive example higher than a")
print(f"random negative example with probability {roc_auc:.2%}")

# ============================================================================
# 6. PRECISION-RECALL CURVES
# ============================================================================

print("\n" + "=" * 80)
print("6. PRECISION-RECALL CURVES")
print("=" * 80)

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
    y_test, y_pred_proba
)

print(f"\nPrecision-Recall Analysis:")
print(f"Max Precision: {precision_vals.max():.4f}")
print(f"Max Recall: {recall_vals.max():.4f}")
print(f"F1-Score at different thresholds:")

for threshold in [0.3, 0.5, 0.7, 0.9]:
    predictions = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(predictions)) > 1:
        f1 = f1_score(y_test, predictions)
        print(f"Threshold {threshold:.1f}: F1={f1:.4f}")

# ============================================================================
# 7. CONFUSION MATRIX AND CLASSIFICATION METRICS
# ============================================================================

print("\n" + "=" * 80)
print("7. CONFUSION MATRIX AND DETAILED METRICS")
print("=" * 80)

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN):  {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP):  {cm[1, 1]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. K-Fold Scores Comparison
ax1 = fig.add_subplot(gs[0, 0])
fold_nums = list(range(1, 6))
ax1.bar(fold_nums, scores, color='steelblue', alpha=0.7, label='Regular K-Fold')
ax1.bar(fold_nums, strat_scores, color='orange', alpha=0.7, label='Stratified K-Fold')
ax1.axhline(y=scores.mean(), color='steelblue', linestyle='--', linewidth=2, label=f'Mean (KFold): {scores.mean():.4f}')
ax1.axhline(y=strat_scores.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean (Stratified): {strat_scores.mean():.4f}')
ax1.set_xlabel('Fold Number')
ax1.set_ylabel('Accuracy Score')
ax1.set_title('K-Fold vs Stratified K-Fold')
ax1.set_xticks(fold_nums)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Multiple Metrics Comparison
ax2 = fig.add_subplot(gs[0, 1])
metrics_names = list(scoring)
metrics_means = [cv_results[f'test_{m}'].mean() for m in scoring]
metrics_stds = [cv_results[f'test_{m}'].std() for m in scoring]
ax2.bar(metrics_names, metrics_means, yerr=metrics_stds, capsize=5, color='steelblue')
ax2.set_ylabel('Score')
ax2.set_title('Cross-Validation Metrics Comparison')
ax2.set_ylim([0, 1.05])
for i, (name, mean) in enumerate(zip(metrics_names, metrics_means)):
    ax2.text(i, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom')
ax2.grid(axis='y', alpha=0.3)

# 3. ROC Curve
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 4. Precision-Recall Curve
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(recall_vals, precision_vals, color='darkgreen', lw=2, label='Precision-Recall Curve')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Confusion Matrix Heatmap
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
ax5.set_ylabel('True Label')
ax5.set_xlabel('Predicted Label')
ax5.set_title('Confusion Matrix')

# 6. Cross-Validation Fold Scores Distribution
ax6 = fig.add_subplot(gs[2, 1])
all_cv_scores = [scores, strat_scores]
labels = ['K-Fold', 'Stratified K-Fold']
ax6.boxplot(all_cv_scores, labels=labels)
ax6.set_ylabel('Accuracy Score')
ax6.set_title('Distribution of CV Scores')
ax6.grid(axis='y', alpha=0.3)

plt.savefig('cross_validation_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("Cross-Validation and Model Evaluation Complete!")
print("=" * 80)
