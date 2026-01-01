"""
Ensemble Methods with Decision Trees

This script demonstrates:
- Bagging (Bootstrap Aggregating)
- Random Forests
- Gradient Boosting
- AdaBoost
- Voting Classifiers
- Stacking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 80)
print("ENSEMBLE METHODS WITH DECISION TREES")
print("=" * 80)

# Load data
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 1. BAGGING
# ============================================================================

print("\n" + "=" * 80)
print("1. BAGGING (Bootstrap Aggregating)")
print("=" * 80)

# Single Decision Tree
single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)
single_acc = single_tree.score(X_test, y_test)

# Bagging
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)
bagging_clf.fit(X_train, y_train)
bagging_acc = bagging_clf.score(X_test, y_test)

print(f"\nSingle Decision Tree Accuracy: {single_acc:.4f}")
print(f"Bagging Accuracy (10 estimators): {bagging_acc:.4f}")
print(f"Improvement: {(bagging_acc - single_acc):.4f}")

# Bagging with different number of estimators
bagging_scores = []
for n in [5, 10, 20, 50, 100]:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=n,
        random_state=42
    )
    bag.fit(X_train, y_train)
    score = bag.score(X_test, y_test)
    bagging_scores.append(score)
    print(f"Bagging with {n:3d} estimators: {score:.4f}")

# ============================================================================
# 2. RANDOM FOREST
# ============================================================================

print("\n" + "=" * 80)
print("2. RANDOM FOREST")
print("=" * 80)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)
rf_acc = rf_clf.score(X_test, y_test)

rf_pred = rf_clf.predict(X_test)
rf_cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)

print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
print(f"Cross-Validation Scores: {rf_cv_scores}")
print(f"Mean CV Score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Feature importance
feature_importance = rf_clf.feature_importances_
top_10_idx = np.argsort(feature_importance)[-10:][::-1]

print(f"\nTop 10 Most Important Features:")
for i, idx in enumerate(top_10_idx, 1):
    print(f"{i:2d}. {X.columns[idx]:30s}: {feature_importance[idx]:.4f}")

# ============================================================================
# 3. GRADIENT BOOSTING
# ============================================================================

print("\n" + "=" * 80)
print("3. GRADIENT BOOSTING")
print("=" * 80)

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_clf.fit(X_train, y_train)
gb_acc = gb_clf.score(X_test, y_test)

gb_pred = gb_clf.predict(X_test)

print(f"\nGradient Boosting Accuracy: {gb_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, gb_pred))

# Feature importance
gb_importance = gb_clf.feature_importances_
top_10_gb_idx = np.argsort(gb_importance)[-10:][::-1]

print(f"\nTop 10 Most Important Features:")
for i, idx in enumerate(top_10_gb_idx, 1):
    print(f"{i:2d}. {X.columns[idx]:30s}: {gb_importance[idx]:.4f}")

# ============================================================================
# 4. ADABOOST
# ============================================================================

print("\n" + "=" * 80)
print("4. ADABOOST")
print("=" * 80)

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_acc = ada_clf.score(X_test, y_test)

ada_pred = ada_clf.predict(X_test)

print(f"\nAdaBoost Accuracy: {ada_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, ada_pred))

# ============================================================================
# 5. VOTING CLASSIFIER
# ============================================================================

print("\n" + "=" * 80)
print("5. VOTING CLASSIFIER (Hard Voting)")
print("=" * 80)

voting_clf = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
    ],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
voting_acc = voting_clf.score(X_test, y_test)
voting_pred = voting_clf.predict(X_test)

print(f"\nVoting Classifier Accuracy: {voting_acc:.4f}")
print(f"\nIndividual Estimator Accuracies:")
for clf_name, clf in voting_clf.estimators_:
    print(f"{clf_name}: {clf.score(X_test, y_test):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, voting_pred))

# ============================================================================
# 6. COMPARISON AND VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("6. MODEL COMPARISON")
print("=" * 80)

models = {
    'Single Tree': single_acc,
    'Bagging': bagging_acc,
    'Random Forest': rf_acc,
    'Gradient Boosting': gb_acc,
    'AdaBoost': ada_acc,
    'Voting': voting_acc
}

print("\nAccuracy Comparison:")
for model_name, accuracy in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {accuracy:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Accuracy Comparison
ax1 = axes[0, 0]
model_names = list(models.keys())
model_accs = list(models.values())
colors = ['red' if m == 'Single Tree' else 'steelblue' for m in model_names]
ax1.bar(model_names, model_accs, color=colors)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylim([min(model_accs) - 0.05, 1.0])
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(model_accs):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# 2. Bagging Estimators Effect
ax2 = axes[0, 1]
ax2.plot([5, 10, 20, 50, 100], bagging_scores, marker='o', linewidth=2, markersize=8)
ax2.axhline(y=single_acc, color='r', linestyle='--', label='Single Tree')
ax2.set_xlabel('Number of Estimators')
ax2.set_ylabel('Accuracy')
ax2.set_title('Bagging: Effect of Number of Estimators')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Random Forest Feature Importance (Top 10)
ax3 = axes[1, 0]
top_features = X.columns[top_10_idx]
top_importances = feature_importance[top_10_idx]
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax3.barh(range(len(top_features)), top_importances, color=colors_imp)
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features)
ax3.set_xlabel('Importance Score')
ax3.set_title('Random Forest: Top 10 Feature Importance')
ax3.invert_yaxis()

# 4. Confusion Matrices Comparison
ax4 = axes[1, 1]
rf_cm = confusion_matrix(y_test, rf_pred)
gb_cm = confusion_matrix(y_test, gb_pred)

print(f"\nRandom Forest Confusion Matrix:")
print(rf_cm)
print(f"\nGradient Boosting Confusion Matrix:")
print(gb_cm)

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
ax4.set_ylabel('True Label')
ax4.set_xlabel('Predicted Label')
ax4.set_title('Random Forest Confusion Matrix')

plt.tight_layout()
plt.savefig('ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("Ensemble Methods Analysis Complete!")
print("=" * 80)
