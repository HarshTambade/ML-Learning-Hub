# Chapter 01 Mini-Projects

## Project 1: Iris Classification - Hello to ML

### Objective
Build your first machine learning model to classify iris flowers based on their physical characteristics.

### Dataset
- **Name**: Iris Dataset
- **Samples**: 150 iris flowers
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Source**: Built-in with scikit-learn

### What You Will Learn
1. Loading and exploring datasets
2. Understanding features and labels
3. Splitting data into training and testing sets
4. Training a simple classifier
5. Making predictions
6. Evaluating model performance

### Steps to Complete the Project

#### Step 1: Load the Data
```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
```

#### Step 2: Explore the Data
- Print dataset shape
- Print sample features
- Understand the target variable
- Check for data imbalance

#### Step 3: Prepare Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 4: Create and Train Model
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

#### Step 5: Make Predictions
```python
predictions = model.predict(X_test)
```

#### Step 6: Evaluate
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

### Key Concepts to Understand
- **Features**: Input variables (X) - the iris flower measurements
- **Target**: Output variable (y) - the species classification
- **Training Set**: Used to teach the model patterns
- **Testing Set**: Used to evaluate how well the model works on new data
- **Accuracy**: Percentage of correct predictions

### Expected Outcomes
- Accuracy should be ~90-95% with Decision Tree
- Understand the ML pipeline workflow
- Gain confidence in building ML models

### Challenges (Optional)
1. Try different algorithms (LogisticRegression, SVM, KNeighborsClassifier)
2. Plot the feature importance
3. Visualize the decision tree
4. Tune hyperparameters to improve accuracy
5. Try different train-test splits (70-30, 80-20)

### Resources
- Scikit-learn documentation: https://scikit-learn.org/
- Iris dataset info: https://en.wikipedia.org/wiki/Iris_flower_data_set

### Success Criteria
- ✅ Code runs without errors
- ✅ Model achieves >85% accuracy
- ✅ Can explain each step in the ML pipeline
- ✅ Understand difference between training and testing data
