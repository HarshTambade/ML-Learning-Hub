"""
Basic SVM Classification Example
Demonstrates SVM for iris flower classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(target_names)}")
print(f"\nFeatures: {feature_names}")
print(f"Target classes: {target_names}\n")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")

# Feature scaling (VERY IMPORTANT for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print(f"Train mean: {X_train_scaled.mean():.6f}")
print(f"Train std: {X_train_scaled.std():.6f}\n")

# Create and train SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
print("Training SVM classifier...")
svm.fit(X_train_scaled, y_train)
print("Training completed!\n")

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Support vectors information
print(f"\nNumber of support vectors: {len(svm.support_)}")
print(f"Percentage of support vectors: {len(svm.support_) / len(X_train) * 100:.2f}%")
print(f"Support vector indices (first 10): {svm.support_[:10]}")

# Decision function margins
decision_function = svm.decision_function(X_test_scaled)
print(f"\nDecision function output shape: {decision_function.shape}")
print(f"Min margin: {decision_function.min():.4f}")
print(f"Max margin: {decision_function.max():.4f}")

# Visualization: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - SVM Iris Classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
plt.show()

# Visualization: Feature space (2D projection using first two features)
plt.figure(figsize=(10, 8))

# Plot training data
for i, target in enumerate(target_names):
    indices = y_train == i
    plt.scatter(X_train_scaled[indices, 0], X_train_scaled[indices, 1],
               label=f'{target} (train)', alpha=0.7, s=100)

# Plot test data with different markers
for i, target in enumerate(target_names):
    indices = y_test == i
    plt.scatter(X_test_scaled[indices, 0], X_test_scaled[indices, 1],
               label=f'{target} (test)', alpha=0.7, s=100, marker='^')

# Highlight support vectors
for i in svm.support_:
    idx_in_train = i
    if idx_in_train < len(X_train_scaled):
        plt.scatter(X_train_scaled[idx_in_train, 0],
                   X_train_scaled[idx_in_train, 1],
                   s=400, linewidth=1.5, edgecolors='red', facecolors='none',
                   label='Support vectors')

plt.xlabel(f'{feature_names[0]} (scaled)')
plt.ylabel(f'{feature_names[1]} (scaled)')
plt.title('SVM Classification - Feature Space (First 2 Features)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_space.png', dpi=100, bbox_inches='tight')
print("Feature space plot saved as 'feature_space.png'")
plt.show()

# Cross-validation scores
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\n" + "="*50)
print("SVM Classification Complete!")
print("="*50)
