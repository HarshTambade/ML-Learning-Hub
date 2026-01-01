"""Decision Tree Classification - Basic Implementation

Demonstrates:
- Building decision trees from scratch
- Using scikit-learn's DecisionTreeClassifier
- Visualizing tree structure
- Making predictions on new data
- Understanding tree parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ============================================================================
# 1. BASIC DECISION TREE WITH IRIS DATASET
# ============================================================================

print("="*70)
print("BASIC DECISION TREE CLASSIFICATION")
print("="*70)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset Shape:")
print(f"  Training samples: {X_train.shape}")
print(f"  Test samples: {X_test.shape}")
print(f"  Features: {feature_names}")
print(f"  Classes: {target_names}")

# Create and train decision tree
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

print(f"\nTree Information:")
print(f"  Tree depth: {dt_classifier.get_depth()}")
print(f"  Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"  Number of features used: {dt_classifier.n_features_in_}")

# Make predictions
y_pred = dt_classifier.predict(X_test)
y_pred_proba = dt_classifier.predict_proba(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
feature_importance = dt_classifier.feature_importances_
print(f"\nFeature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"  {name}: {importance:.4f}")

# ============================================================================
# 2. TREE VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("TREE VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot 1: Decision Tree
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          ax=axes[0],
          fontsize=10)
axes[0].set_title("Decision Tree - Iris Dataset", fontsize=14, fontweight='bold')

# Plot 2: Feature Importance
axes[1].barh(feature_names, feature_importance, color='steelblue')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('decision_tree_basic.png', dpi=100, bbox_inches='tight')
print("\nTree visualization saved as 'decision_tree_basic.png'")
plt.show()

# ============================================================================
# 3. PREDICT NEW SAMPLES
# ============================================================================

print("\n" + "="*70)
print("PREDICTION ON NEW SAMPLES")
print("="*70)

# Sample new flowers
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
    [7.0, 3.2, 4.7, 1.4],  # Likely Versicolor
    [6.3, 3.3, 6.0, 2.5],  # Likely Virginica
])

predictions = dt_classifier.predict(new_samples)
probabilities = dt_classifier.predict_proba(new_samples)

print("\nNew Sample Predictions:")
for i, (sample, pred, proba) in enumerate(zip(new_samples, predictions, probabilities)):
    print(f"\n  Sample {i+1}: {sample}")
    print(f"  Predicted class: {target_names[pred]}")
    print(f"  Probabilities:")
    for cls, prob in zip(target_names, proba):
        print(f"    {cls}: {prob:.4f}")

# ============================================================================
# 4. CONFUSION MATRIX
# ============================================================================

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_title('Confusion Matrix - Decision Tree', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_dt.png', dpi=100, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix_dt.png'")
plt.show()

# ============================================================================
# 5. PARAMETER TUNING EXPLORATION
# ============================================================================

print("\n" + "="*70)
print("PARAMETER TUNING EXPLORATION")
print("="*70)

# Test different max_depth values
max_depths = [1, 2, 3, 5, 10, None]
accuracies_train = []
accuracies_test = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    
    accuracies_train.append(train_acc)
    accuracies_test.append(test_acc)
    
    print(f"  max_depth={depth}: Train={train_acc:.4f}, Test={test_acc:.4f}")

# Plot learning curve
fig, ax = plt.subplots(figsize=(10, 6))
depth_labels = [str(d) if d else 'None' for d in max_depths]
ax.plot(depth_labels, accuracies_train, 'o-', label='Training Accuracy', linewidth=2)
ax.plot(depth_labels, accuracies_test, 's-', label='Test Accuracy', linewidth=2)
ax.set_xlabel('Max Depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Decision Tree Accuracy vs Tree Depth', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_tuning.png', dpi=100, bbox_inches='tight')
print("\nDepth tuning plot saved as 'depth_tuning.png'")
plt.show()

print("\n" + "="*70)
print("BASIC EXAMPLE COMPLETE")
print("="*70)
