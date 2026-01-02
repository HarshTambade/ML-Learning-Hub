"""
Multi-Class SVM - One-vs-Rest vs One-vs-One
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Multi-Class SVM Strategies\n" + "="*50)

# One-vs-Rest (OvR) - Default in scikit-learn
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
svm_ovr.fit(X_train, y_train)
ovr_train = svm_ovr.score(X_train, y_train)
ovr_test = svm_ovr.score(X_test, y_test)

print(f"\nOne-vs-Rest (OvR):")
print(f"  Train accuracy: {ovr_train:.4f}")
print(f"  Test accuracy:  {ovr_test:.4f}")
print(f"  Number of classifiers: {len(target_names)}")
print(f"  Prediction time: O(k*n)")

# One-vs-One (OvO)
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)
ovo_train = svm_ovo.score(X_train, y_train)
ovo_test = svm_ovo.score(X_test, y_test)

print(f"\nOne-vs-One (OvO):")
print(f"  Train accuracy: {ovo_train:.4f}")
print(f"  Test accuracy:  {ovo_test:.4f}")
print(f"  Number of classifiers: {len(target_names) * (len(target_names) - 1) // 2}")
print(f"  Prediction time: O(k^2)")

# Predictions and evaluation
y_pred_ovr = svm_ovr.predict(X_test)
y_pred_ovo = svm_ovo.predict(X_test)

print(f"\n\nClassification Reports:")
print(f"\nOvR Strategy:")
print(classification_report(y_test, y_pred_ovr, target_names=target_names))

print(f"\nOvO Strategy:")
print(classification_report(y_test, y_pred_ovo, target_names=target_names))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_ovr = confusion_matrix(y_test, y_pred_ovr)
sns.heatmap(cm_ovr, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=target_names, yticklabels=target_names)
axes[0].set_title('One-vs-Rest Confusion Matrix')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')

cm_ovo = confusion_matrix(y_test, y_pred_ovo)
sns.heatmap(cm_ovo, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=target_names, yticklabels=target_names)
axes[1].set_title('One-vs-One Confusion Matrix')
axes[1].set_ylabel('True label')
axes[1].set_xlabel('Predicted label')

plt.tight_layout()
plt.savefig('multi_class_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("Comparison Summary:")
print(f"OvR better for: Few classes, interpretability")
print(f"OvO better for: Many classes, more expensive training")
print("="*50)
