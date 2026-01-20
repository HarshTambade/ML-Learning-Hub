"""Cross-Validation Techniques

Implementation of k-fold, stratified k-fold, and other CV methods
for robust model evaluation.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, LeaveOneOut
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = load_iris()
X, y = data.data, data.target

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier()
}

cv_methods = {
    'K-Fold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified K-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Leave-One-Out': LeaveOneOut()
}

results = {}

for model_name, model in models.items():
    results[model_name] = {}
    print(f"\n{model_name}:")
    
    for cv_name, cv in cv_methods.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[model_name][cv_name] = scores
        print(f"{cv_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

fig, ax = plt.subplots(figsize=(10, 6))

for model_name, cv_results in results.items():
    means = [cv_results[cv_name].mean() for cv_name in cv_methods.keys()]
    ax.plot(list(cv_methods.keys()), means, marker='o', label=model_name)

ax.set_ylabel('Accuracy')
ax.set_title('Cross-Validation Results')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
