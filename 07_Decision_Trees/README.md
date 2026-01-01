# Chapter 07: Decision Trees

## 🌳 Introduction

Decision Trees are one of the most intuitive and interpretable machine learning algorithms. They work by recursively partitioning the feature space based on feature values to make predictions. Perfect for both classification and regression tasks!

## 📊 Decision Tree Structure Diagram

```
                          Root Node
                      (Best Split Feature)
                             |
              _______________|_______________
             /                               \
         [Feature X <= 5.5]            [Feature X > 5.5]
          /                                     \
       Node 2                                 Node 3
      /      \                               /      \
   [Y<=3]  [Y>3]                        [Z<=2]  [Z>2]
   /   \    /  \                        /  \     /  \
 Leaf  Leaf Leaf Leaf              Leaf  Leaf Leaf Leaf
 Class Class Class Class           Class Class Class Class
  0     1    1     2                1     2    2     2
```

## Key Characteristics

✅ **Highly Interpretable** - Easy to visualize and understand
✅ **No Feature Scaling** - Works with raw features
✅ **Handles Non-linear** - Captures complex relationships
✅ **Fast Predictions** - O(log n) complexity
✅ **Feature Importance** - Automatic importance calculation
✅ **Mixed Data Types** - Works with numeric and categorical

❌ **Prone to Overfitting** - Can memorize training data
❌ **Unstable** - Small data changes → big tree changes
❌ **Biased Data** - May favor dominant classes
❌ **High Variance** - Sensitive to data variations

## How Decision Trees Work

### 1. **Recursive Partitioning**
```
Algorithm:
1. Start with all samples in root
2. For each node:
   - Find best feature to split on
   - Split samples into left and right
   - Recursively repeat on child nodes
   - Stop when: purity threshold OR max depth OR min samples
```

### 2. **Split Quality Metrics**

#### Information Gain (Entropy)
```
IGain = Entropy(parent) - Σ(weight × Entropy(child))

Where:
  Entropy(S) = -Σ p_i * log₂(p_i)
  p_i = proportion of class i
```

#### Gini Impurity
```
Gini(S) = 1 - Σ p_i²

Gini_Gain = Gini(parent) - Σ(weight × Gini(child))
```

## Chapter Contents

### 📁 **code_examples/**
1. `01_basic_decision_tree.py` - Basic classification with Iris dataset
   - Tree visualization
   - Feature importance analysis
   - Parameter tuning
   - Confusion matrix

2. `02_regression_tree.py` - Decision tree for regression
3. `03_feature_engineering.py` - Handling categorical features
4. `04_pruning_techniques.py` - Preventing overfitting
5. `05_tree_comparison.py` - ID3 vs CART vs C4.5 algorithms

### 📚 **notes/**
1. `01_tree_fundamentals.md` - Basic concepts and mathematics
2. `02_splitting_criteria.md` - Information Gain and Gini Impurity
3. `03_entropy_calculation.md` - Step-by-step entropy examples
4. `04_tree_visualization.md` - Understanding tree structure
5. `05_overfitting_control.md` - Pruning and regularization

### 🏋️ **exercises/**
- Building decision trees from scratch
- Calculating entropy and information gain
- Comparing different impurity measures
- Visualizing splitting decisions
- Handling missing values
- Imbalanced dataset problems

**30+ YouTube Video Resources** from:
- StatQuest with Josh Starmer
- Andrew Ng (ML course)
- Jeremy Howard (Fast.ai)
- Krish Naik
- And more...

### 🚀 **projects/**
1. **Titanic Survival Prediction** - Classic classification
2. **House Price Prediction** - Regression task
3. **Credit Approval** - Binary classification
4. **Customer Segmentation** - Multi-class problem
5. **Medical Diagnosis** - Healthcare application
6. **Loan Default** - Financial prediction
7. **Iris Species** - Multi-class classification
8. **Employee Churn** - Retention prediction

## Learning Path

### 🟢 Beginner (Hours 1-3)
1. Understand basic tree structure
2. Learn entropy and information gain
3. Run `01_basic_decision_tree.py`
4. Visualize a simple tree
5. Make predictions on new data

### 🟡 Intermediate (Hours 4-8)
1. Study splitting criteria (Gini vs Entropy)
2. Implement tree from scratch
3. Learn about overfitting and pruning
4. Compare different algorithms
5. Work on projects

### 🔴 Advanced (Hours 9-15)
1. Master parameter tuning
2. Implement cost-complexity pruning
3. Handle missing values
4. Deal with imbalanced data
5. Optimize for production

## Quick Start Code

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create and train tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Predict
predictions = tree.predict(X_test)
accuracy = tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
for feat, imp in zip(iris.feature_names, tree.feature_importances_):
    print(f"{feat}: {imp:.4f}")
```

## Key Parameters

### `max_depth`
- Controls tree height
- Smaller = simpler (prevent overfitting)
- Larger = complex (capture patterns)
- Default: None (unlimited)

### `min_samples_split`
- Minimum samples to split a node
- Prevents small groups
- Default: 2
- Larger values = simpler tree

### `min_samples_leaf`
- Minimum samples in leaf node
- Prevents single-instance leaves
- Default: 1
- Higher values reduce overfitting

### `criterion`
- "gini" - Gini impurity (default)
- "entropy" - Information gain
- Minimal difference in most cases

## Real-World Applications

```
🏥 Healthcare
  ├─ Diagnosis prediction
  ├─ Treatment recommendation
  └─ Risk assessment

💰 Finance
  ├─ Loan approval
  ├─ Credit scoring
  └─ Fraud detection

🛒 E-commerce
  ├─ Customer segmentation
  ├─ Product recommendation
  └─ Churn prediction

📱 Technology
  ├─ Feature selection
  ├─ Bug prediction
  └─ User behavior analysis
```

## Advantages vs Disadvantages

### ✅ Advantages
- Interpretable and visual
- No feature scaling needed
- Handles mixed data types
- Non-parametric
- Fast inference
- Feature importance built-in

### ❌ Disadvantages
- Prone to overfitting
- High variance
- Biased toward dominant classes
- Unstable (small changes → big impact)
- Greedy algorithm (not globally optimal)

## Common Challenges

### Challenge 1: Overfitting
**Problem**: Tree memorizes training data
**Solutions**:
- Limit max_depth
- Increase min_samples_split
- Increase min_samples_leaf
- Use pruning
- Use ensemble (Random Forest, Gradient Boosting)

### Challenge 2: Imbalanced Data
**Problem**: Biased toward majority class
**Solutions**:
- Use class_weight='balanced'
- Adjust sample_weight
- Use SMOTE/Oversampling
- Evaluate with F1, not accuracy

### Challenge 3: Categorical Features
**Problem**: Decision trees split on values, not categories
**Solutions**:
- One-hot encoding
- Label encoding (for ordinal)
- Use algorithms supporting categories

## Visualization & Diagrams

### Tree Growing Process
```
Iteration 1: Pure leaf
     [Class A: 10, B: 0]

Iteration 2: First split
        Root
       /    \
    [A:7]  [A:3, B:5]
   (Pure) (Mixed)

Iteration 3-N: Continue splitting impure nodes
        Root
       /    \
    [A:7]   Node2
   (Pure)  /    \
        [B:4]  [A:3,B:1]
        (Pure) (Mixed)
```

## Performance Metrics

- **Accuracy**: (TP+TN)/(Total) - overall correctness
- **Precision**: TP/(TP+FP) - positive prediction quality
- **Recall**: TP/(TP+FN) - positive detection rate
- **F1-Score**: 2×(Prec×Rec)/(Prec+Rec) - harmonic mean
- **ROC-AUC**: Area under ROC curve - threshold independence

## Resource Links

### Recommended YouTube Channels
- StatQuest Decision Trees: [Watch](https://www.youtube.com/watch?v=7VeUAPc2hdM)
- Andrew Ng Machine Learning: [Watch](https://www.coursera.org/learn/machine-learning)
- Jeremy Howard Fast.ai: [Watch](https://course.fast.ai/)
- Krish Naik: [Watch](https://www.youtube.com/@krishnaik06)

### Books
- ISLR: Chapters 8.1-8.2
- Elements of Statistical Learning: Chapter 9
- Pattern Recognition: Chapter 14

### Datasets for Practice
- Iris (built-in): Flower classification
- Titanic: Kaggle dataset
- Wine: scikit-learn dataset
- Breast Cancer: UCI ML repository
- Adult: Income prediction

## Chapter Statistics

📊 **Content Summary**:
- 5 Code Examples with visualizations
- 5 Detailed Notes Files
- 30+ YouTube Video Links
- 8 Real-World Projects
- 20+ Practice Exercises
- **Total Content**: 15-20 hours of learning

## Next Steps

After mastering Decision Trees:
1. **Random Forests** - Ensemble of trees
2. **Gradient Boosting** - Sequential tree improvement
3. **XGBoost** - Optimized gradient boosting
4. **LightGBM** - Fast boosting
5. **CatBoost** - Categorical boosting

## Tips for Success

1. **Visualize everything** - Draw trees, see splits
2. **Understand entropy** - Key to algorithm
3. **Experiment with parameters** - See their effects
4. **Compare to logistic regression** - Understand differences
5. **Build from scratch** - Deep understanding
6. **Use ensemble methods** - Better performance
7. **Monitor overfitting** - Always validate
8. **Feature engineering** - Better splits

---

**Last Updated**: 2024-01-01
**Difficulty**: Beginner to Intermediate
**Estimated Time**: 15-20 hours
**Prerequisites**: Python, NumPy, Pandas, Scikit-learn
**Next Chapter**: Chapter 08 - Support Vector Machines
