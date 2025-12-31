"""Feature Scaling Impact on Logistic Regression

This script demonstrates how feature scaling affects logistic regression
performance with different scaling techniques.

Author: ML Learning Hub
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. CREATE DATASET WITH DIFFERENT FEATURE SCALES
# ============================================================================

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42
)

# Create different scales manually
X_varied_scale = X.copy()
for i in range(X_varied_scale.shape[1]):
    X_varied_scale[:, i] = X_varied_scale[:, i] * (10 ** i)

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_varied_scale, y, test_size=0.2, random_state=42
)

print("Dataset Info:")
print(f"Original Feature Ranges:")
for i in range(X_train_orig.shape[1]):
    print(f"  Feature {i}: [{X_train_orig[:, i].min():.2e}, {X_train_orig[:, i].max():.2e}]")
print()

# ============================================================================
# 2. DIFFERENT SCALING TECHNIQUES
# ============================================================================

scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer()
}

results = {}

for scaler_name, scaler in scalers.items():
    if scaler is None:
        X_train = X_train_orig.copy()
        X_test = X_test_orig.copy()
    else:
        X_train = scaler.fit_transform(X_train_orig)
        X_test = scaler.transform(X_test_orig)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[scaler_name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"{scaler_name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()

# ============================================================================
# 3. CONVERGENCE SPEED WITH DIFFERENT SCALERS
# ============================================================================

print("\nConvergence Analysis:")
for scaler_name, scaler in scalers.items():
    if scaler is None:
        X_train = X_train_orig.copy()
    else:
        X_train = scaler.fit_transform(X_train_orig)
    
    model = LogisticRegression(
        max_iter=5000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print(f"{scaler_name}: n_iter = {model.n_iter_[0] if hasattr(model, 'n_iter_') else 'N/A'}")

print()

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Scaling Impact on Logistic Regression', fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
scaler_names = list(results.keys())
train_accs = [results[name]['train_acc'] for name in scaler_names]
test_accs = [results[name]['test_acc'] for name in scaler_names]

x = np.arange(len(scaler_names))
width = 0.35
ax1.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy with Different Scalers')
ax1.set_xticks(x)
ax1.set_xticklabels(scaler_names, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim([0.8, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Cross-Validation Scores
ax2 = axes[0, 1]
cv_means = [results[name]['cv_mean'] for name in scaler_names]
cv_stds = [results[name]['cv_std'] for name in scaler_names]
ax2.bar(scaler_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
ax2.set_ylabel('CV Accuracy')
ax2.set_title('Cross-Validation Scores')
ax2.set_xticklabels(scaler_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature Scale Comparison (unscaled vs scaled)
ax3 = axes[1, 0]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_orig)
feature_idx = [0, 1, 5, 9]
for idx in feature_idx:
    ax3.hist(X_train_orig[:, idx], alpha=0.5, label=f'Feature {idx} (Original)')
ax3.set_xlabel('Feature Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Original Feature Scales (Highly Varied)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Scaled Feature Comparison
ax4 = axes[1, 1]
for idx in feature_idx:
    ax4.hist(X_train_scaled[:, idx], alpha=0.5, label=f'Feature {idx} (Scaled)')
ax4.set_xlabel('Feature Value')
ax4.set_ylabel('Frequency')
ax4.set_title('Scaled Features (Standardized)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Feature Scaling Analysis Complete!")
print("="*70)
