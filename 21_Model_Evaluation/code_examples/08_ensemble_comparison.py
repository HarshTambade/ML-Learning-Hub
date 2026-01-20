"""Ensemble Model Comparison"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()

df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
df = df.sort_values('Accuracy', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(df['Model'], df['Accuracy'])
plt.xlabel('Cross-Validation Accuracy')
plt.title('Model Comparison')
plt.xlim([0.9, 1.0])
plt.show()

print(df.to_string())
