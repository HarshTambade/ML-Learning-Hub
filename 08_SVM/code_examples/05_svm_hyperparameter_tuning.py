"""
SVM Hyperparameter Tuning
Optimize C and gamma using GridSearchCV
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("SVM Hyperparameter Tuning\n" + "="*60)
print(f"Dataset: Breast Cancer")
print(f"Samples: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
print(f"Features: {X.shape[1]}\n")

# GridSearchCV with parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

print(f"Grid Search Space: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinations\n")

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("Tuning hyperparameters (this may take a moment)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Test on held-out test set
best_svm = grid_search.best_estimator_
test_score = best_svm.score(X_test_scaled, y_test)
print(f"Test set accuracy: {test_score:.4f}")

y_pred = best_svm.predict(X_test_scaled)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize parameter importance
results_df = pd.DataFrame(grid_search.cv_results_)

# Plot C vs Mean Test Score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

C_values = param_grid['C']
mean_scores_rbf = []
mean_scores_linear = []

for c in C_values:
    rbf_rows = results_df[(results_df['param_C'] == c) & (results_df['param_kernel'] == 'rbf')]
    linear_rows = results_df[(results_df['param_C'] == c) & (results_df['param_kernel'] == 'linear')]
    
    mean_scores_rbf.append(rbf_rows['mean_test_score'].mean())
    mean_scores_linear.append(linear_rows['mean_test_score'].mean())

axes[0].plot(C_values, mean_scores_rbf, marker='o', label='RBF kernel', linewidth=2)
axes[0].plot(C_values, mean_scores_linear, marker='s', label='Linear kernel', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_xlabel('C (Regularization Parameter)')
axes[0].set_ylabel('Mean Cross-Validation Accuracy')
axes[0].set_title('Effect of C Parameter')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Gamma vs Mean Test Score (RBF only)
gamma_values = [g for g in param_grid['gamma'] if isinstance(g, float)]
rbf_gamma_scores = []

for g in gamma_values:
    gamma_rows = results_df[(results_df['param_gamma'] == g) & (results_df['param_kernel'] == 'rbf')]
    rbf_gamma_scores.append(gamma_rows['mean_test_score'].mean())

axes[1].plot(gamma_values, rbf_gamma_scores, marker='o', linewidth=2, color='green')
axes[1].set_xscale('log')
axes[1].set_xlabel('Gamma (Kernel Coefficient)')
axes[1].set_ylabel('Mean Cross-Validation Accuracy')
axes[1].set_title('Effect of Gamma Parameter (RBF Kernel)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_tuning.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Tuning Summary:")
print("- Best kernel: RBF for non-linear separation")
print("- C parameter: Controls margin vs misclassification tradeoff")
print("- Gamma parameter: Controls decision boundary smoothness")
print("="*60)
