# Chapter 07: Decision Trees - Exercises

## 📚 Practice Problems with Solutions

### Exercise Set 1: Understanding Tree Fundamentals

#### 1.1 Tree Construction from Data
**Problem**: Build a simple decision tree manually on paper using the following data:

```
Feature1 | Feature2 | Class
   1     |    2     |   A
   2     |    3     |   A
   3     |    4     |   B
   4     |    5     |   B
```

**Task**:
- Calculate entropy at the root
- Find the best feature to split on
- Show the resulting tree structure

**Solution Approach**:
1. Entropy(root) = -0.5*log2(0.5) -0.5*log2(0.5) = 1.0
2. Try splits on Feature1 and Feature2
3. Calculate information gain for each
4. Choose the split with highest gain

#### 1.2 Gini vs Entropy
**Problem**: Compare Gini impurity and entropy for classification.

**Task**:
- Calculate both metrics for a node with [60% Class A, 40% Class B]
- Explain the difference
- When would you use each?

**Solution**:
- Entropy: -0.6*log2(0.6) - 0.4*log2(0.4) = 0.971
- Gini: 1 - (0.6² + 0.4²) = 0.48
- Use entropy for information gain; Gini for CART algorithm

### Exercise Set 2: Parameter Tuning

#### 2.1 Max Depth Impact
**Problem**: Create trees with different max_depth values on the Iris dataset and observe overfitting.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    
    print(f"Depth={depth}: Train={train_acc:.3f}, Test={test_acc:.3f}")
```

**Task**:
- Run this code
- Plot training vs test accuracy
- Identify the optimal depth
- Explain why deeper trees don't always generalize better

#### 2.2 Min Samples Parameters
**Problem**: Experiment with min_samples_split and min_samples_leaf.

**Task**:
- Test values: 2, 5, 10, 20, 50
- Measure impact on tree size and accuracy
- Create a table comparing results
- Recommend optimal values for your data

### Exercise Set 3: Feature Engineering

#### 3.1 Categorical Encoding
**Problem**: Work with categorical features in decision trees.

**Task**:
- Load a dataset with categorical variables
- Try different encoding methods:
  - One-hot encoding
  - Label encoding
- Compare decision tree performance
- Which encoding works better? Why?

#### 3.2 Feature Importance Analysis
**Problem**: Analyze which features matter most.

```python
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

for feat, imp in zip(feature_names, tree.feature_importances_):
    if imp > 0:
        print(f"{feat}: {imp:.4f}")
```

**Task**:
- Identify top 3 most important features
- Remove low-importance features and retrain
- Does model performance improve?
- Create a visualization

### Exercise Set 4: Handling Missing Values

#### 4.1 Missing Data Strategies
**Problem**: Decision trees handle missing values differently.

**Task**:
- Introduce missing values (NaN) to your data
- Try different handling strategies:
  - Remove rows with missing values
  - Fill with mean/mode
  - Use surrogate splits
- Compare results
- Which strategy preserves performance best?

### Exercise Set 5: Imbalanced Datasets

#### 5.1 Class Imbalance Problem
**Problem**: Tree biases toward dominant class.

**Task**:
- Create imbalanced dataset (70% Class A, 30% Class B)
- Train tree without correction
- Train tree with class_weight='balanced'
- Compare:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Which metric matters most?

### Exercise Set 6: Overfitting Prevention

#### 6.1 Pruning
**Problem**: Use cost complexity pruning to prevent overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Cost complexity pruning
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

for alpha in ccp_alphas[-5:]:
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    pruned_tree.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, pruned_tree.predict(X_train))
    test_acc = accuracy_score(y_test, pruned_tree.predict(X_test))
    
    print(f"Alpha={alpha:.4f}: Train={train_acc:.3f}, Test={test_acc:.3f}")
```

**Task**:
- Identify best alpha value
- Compare pruned vs unpruned tree
- Visualize both trees
- Note difference in complexity

### Exercise Set 7: Visualization

#### 7.1 Tree Visualization
**Problem**: Understand tree structure visually.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=class_names,
          filled=True, ax=ax, fontsize=10)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=100)
plt.show()
```

**Task**:
- Generate tree visualization
- Trace a single prediction path
- Identify where misclassifications occur
- Explain tree decisions in plain English

### Exercise Set 8: Real-World Application

#### 8.1 Titanic Survival Prediction
**Problem**: Predict Titanic passenger survival using decision trees.

**Task**:
1. Load Titanic dataset
2. Preprocess features
3. Train decision tree
4. Achieve >80% accuracy
5. Analyze which features matter most
6. Interpret important decision rules
7. Compare to other models

## 🎬 YouTube Video Resources (30+ Videos)

### Core Concepts
- StatQuest Decision Trees Playlist: https://www.youtube.com/watch?v=7VeUAPc2hdM
- Andrew Ng Trees: https://www.coursera.org/learn/machine-learning
- Jeremy Howard Fast.ai Trees: https://course.fast.ai/

### Advanced Topics
- Pruning and Regularization: https://www.youtube.com/results?search_query=decision+tree+pruning
- Feature Importance: https://www.youtube.com/results?search_query=decision+tree+feature+importance
- Categorical Data: https://www.youtube.com/results?search_query=decision+tree+categorical

## 💡 Tips for Solving Exercises

1. **Start Simple**: Begin with small datasets before real-world data
2. **Visualize Everything**: Draw trees, plot accuracies, show splits
3. **Compare Methods**: Always test alternatives
4. **Document Findings**: Write explanations of results
5. **Validate Results**: Use cross-validation
6. **Real Data**: Apply to actual datasets

## ✅ Self-Assessment

After completing these exercises, you should be able to:
- [ ] Build decision trees from scratch
- [ ] Calculate entropy and Gini impurity
- [ ] Tune tree parameters effectively
- [ ] Handle missing and categorical data
- [ ] Deal with imbalanced datasets
- [ ] Prevent overfitting via pruning
- [ ] Interpret and visualize tree decisions
- [ ] Apply to real-world problems

---

**Level**: Beginner to Intermediate
**Time**: 10-15 hours
**Difficulty**: Moderate
