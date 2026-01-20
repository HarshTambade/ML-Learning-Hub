"""Feature Importance Analysis"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, (name, model) in enumerate([('Random Forest', rf), ('Gradient Boosting', gb)]):
    importances = model.feature_importances_
    indices = pd.Series(importances, index=data.feature_names).sort_values()
    axes[idx].barh(indices.index, indices.values)
    axes[idx].set_title(f'{name} - Feature Importance')
    axes[idx].set_xlabel('Importance')

plt.tight_layout()
plt.show()

print("Random Forest Importances:")
for name, importance in zip(data.feature_names, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")
