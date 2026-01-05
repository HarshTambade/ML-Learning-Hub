"""Feature Selection Methods for Dimensionality Reduction"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create synthetic high-dimensional dataset
X_synthetic, y_synthetic = make_classification(
    n_samples=300, n_features=20, n_informative=8,
    n_redundant=5, n_clusters_per_class=2, random_state=42
)

print("Feature Selection Methods")
print("="*60)

# 1. SelectKBest with f_classif
print("\n1. SelectKBest (Univariate Statistical Tests)")
print("-" * 60)

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

scores = selector.scores_
scores_df = pd.DataFrame({
    'Feature': feature_names,
    'Score': scores
}).sort_values('Score', ascending=False)

print("\nFeature Importance Scores:")
print(scores_df)

print(f"\nSelected features: {np.array(feature_names)[selector.get_support()]}")

# 2. Recursive Feature Elimination (RFE)
print("\n2. Recursive Feature Elimination (RFE)")
print("-" * 60)

estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=2, step=1)
X_rfe = rfe.fit_transform(X, y)

rfe_df = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': rfe.ranking_,
    'Selected': rfe.support_
}).sort_values('Ranking')

print("\nRFE Rankings:")
print(rfe_df)

print(f"\nSelected features: {np.array(feature_names)[rfe.support_]}")

# 3. Feature Selection using Model-based Method
print("\n3. Model-based Feature Selection (Random Forest)")
print("-" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance_df)

selector_model = SelectFromModel(rf_model, prefit=True, max_features=2)
X_model = selector_model.transform(X)
print(f"\nSelected features: {np.array(feature_names)[selector_model.get_support()]}")

# 4. Feature Selection on High-dimensional Data
print("\n4. Feature Selection on High-dimensional Data")
print("-" * 60)

print(f"\nOriginal synthetic data shape: {X_synthetic.shape}")

selector_k = SelectKBest(f_classif, k=8)
X_selected_synth = selector_k.fit_transform(X_synthetic, y_synthetic)
print(f"Selected data shape (k=8): {X_selected_synth.shape}")

rfe_synth = RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
                 n_features_to_select=8, step=1)
X_rfe_synth = rfe_synth.fit_transform(X_synthetic, y_synthetic)
print(f"RFE selected data shape (k=8): {X_rfe_synth.shape}")

# 5. Comparison of Methods
print("\n5. Model Performance with Different Feature Selection Methods")
print("-" * 60)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)

# No feature selection
scores_none = cross_val_score(rf_classifier, X_synthetic, y_synthetic, cv=5)
print(f"\nNo Feature Selection (20 features)")
print(f"  Mean CV Score: {scores_none.mean():.4f} (+/- {scores_none.std():.4f})")

# SelectKBest
pipeline_kbest = Pipeline([
    ('selector', SelectKBest(f_classif, k=8)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])
scores_kbest = cross_val_score(pipeline_kbest, X_synthetic, y_synthetic, cv=5)
print(f"\nSelectKBest (8 features)")
print(f"  Mean CV Score: {scores_kbest.mean():.4f} (+/- {scores_kbest.std():.4f})")

# RFE
pipeline_rfe = Pipeline([
    ('selector', RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
                     n_features_to_select=8, step=1)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])
scores_rfe = cross_val_score(pipeline_rfe, X_synthetic, y_synthetic, cv=5)
print(f"\nRecursive Feature Elimination (8 features)")
print(f"  Mean CV Score: {scores_rfe.mean():.4f} (+/- {scores_rfe.std():.4f})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Feature importance from SelectKBest
ax1 = axes[0, 0]
ax1.barh(range(len(scores_df)), scores_df['Score'])
ax1.set_yticks(range(len(scores_df)))
ax1.set_yticklabels(scores_df['Feature'])
ax1.set_xlabel('Score')
ax1.set_title('SelectKBest - Feature Scores')
ax1.invert_yaxis()

# Feature importance from Random Forest
ax2 = axes[0, 1]
ax2.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
ax2.set_yticks(range(len(feature_importance_df)))
ax2.set_yticklabels(feature_importance_df['Feature'])
ax2.set_xlabel('Importance')
ax2.set_title('Random Forest - Feature Importance')
ax2.invert_yaxis()

# Cross-validation scores comparison
ax3 = axes[1, 0]
methods = ['No Selection\n(20 feat)', 'SelectKBest\n(8 feat)', 'RFE\n(8 feat)']
mean_scores = [scores_none.mean(), scores_kbest.mean(), scores_rfe.mean()]
std_scores = [scores_none.std(), scores_kbest.std(), scores_rfe.std()]
ax3.bar(methods, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
ax3.set_ylabel('Cross-validation Score')
ax3.set_title('Model Performance Comparison')
ax3.set_ylim([0.8, 1.0])

# Feature count vs score trade-off
ax4 = axes[1, 1]
feature_counts = [5, 8, 10, 15, 20]
rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
scores_list = []

for k in feature_counts:
    if k < X_synthetic.shape[1]:
        selector_temp = SelectKBest(f_classif, k=k)
        X_temp = selector_temp.fit_transform(X_synthetic, y_synthetic)
        pipeline_temp = Pipeline([
            ('selector', SelectKBest(f_classif, k=k)),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        score = cross_val_score(pipeline_temp, X_synthetic, y_synthetic, cv=3).mean()
        scores_list.append(score)
    else:
        score = cross_val_score(rf_classifier, X_synthetic, y_synthetic, cv=3).mean()
        scores_list.append(score)

ax4.plot(feature_counts, scores_list, 'o-', linewidth=2, markersize=8)
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Mean CV Score')
ax4.set_title('Features vs Model Performance')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Feature selection analysis complete!")
print("Visualization saved as 'feature_selection_analysis.png'")
