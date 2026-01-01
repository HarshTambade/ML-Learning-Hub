# 05: Overfitting Control and Pruning

## The Overfitting Problem

### What is Overfitting?

Overfitting occurs when a decision tree learns the **specific details and noise** in the training data rather than the underlying patterns. The tree becomes too complex and fits the training data perfectly but fails on new data.

### Symptoms of Overfitting

```
Training Accuracy:  95% (very high)
Test Accuracy:      65% (very low)
                    Large gap = Overfitting!

Train Loss:  0.01 (very low)
Test Loss:   0.45 (high)
```

### Why Trees Overfit

1. **No regularization by default**: Trees can grow indefinitely
2. **Greedy algorithm**: Focuses on training data fits
3. **Noise sensitivity**: Treats noise as signal
4. **High variance**: Small training changes → large tree changes
5. **Complex patterns**: Captures meaningless patterns

## Prevention Strategies

### Strategy 1: Limit Tree Depth

**Parameter**: `max_depth`

```python
from sklearn.tree import DecisionTreeClassifier

# Shallow tree - less overfitting
dt = DecisionTreeClassifier(max_depth=3)

# Deeper tree - more overfitting
dt = DecisionTreeClassifier(max_depth=10)
```

**Effect**:
- Lower max_depth → Simpler tree, less overfitting
- Higher max_depth → Complex tree, more overfitting

**Best Practice**: 
- Start with small depth (3-5)
- Increase gradually while monitoring validation accuracy

### Strategy 2: Minimum Samples to Split

**Parameter**: `min_samples_split`

Minimum number of samples required to split an internal node.

```python
# Prevent splitting on small groups
dt = DecisionTreeClassifier(min_samples_split=20)
```

**Effect**:
- Higher value → Larger leaf nodes → Less specific rules
- Lower value → Smaller leaf nodes → More specific rules

**Benefits**:
- Prevents learning from small, noisy subsets
- Reduces tree complexity

**Default**: 2 (even 1 sample can create a split)

### Strategy 3: Minimum Samples in Leaf

**Parameter**: `min_samples_leaf`

Minimum number of samples required at a leaf node.

```python
# Ensure leaves have at least 5 samples
dt = DecisionTreeClassifier(min_samples_leaf=5)
```

**Effect**:
- Higher value → Larger leaves → Smoother predictions
- Prevents single-sample leaves

**Benefits**:
- Eliminates overly specific rules
- Ensures statistical significance

**Default**: 1

### Strategy 4: Minimum Impurity Decrease

**Parameter**: `min_impurity_decrease`

Only split if impurity decrease is above threshold.

```python
# Only split if it improves purity by at least 0.01
dt = DecisionTreeClassifier(min_impurity_decrease=0.01)
```

**Benefits**:
- Skips marginal splits
- Focuses on significant improvements

### Strategy 5: Maximum Leaf Nodes

**Parameter**: `max_leaf_nodes`

Limits total number of leaf nodes.

```python
# Maximum 10 leaf nodes
dt = DecisionTreeClassifier(max_leaf_nodes=10)
```

**Benefits**:
- Direct control over tree complexity
- Grows best tree up to limit

## Pruning Techniques

Pruning removes branches that don't improve validation accuracy.

### Cost-Complexity Pruning (scikit-learn)

**Concept**: Remove branches with highest complexity vs. accuracy trade-off.

```python
from sklearn.tree import DecisionTreeClassifier

# Create full tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Get cost-complexity pruning path
path = dt.cost_complexity_pruning_path(X_val, y_val)
alphas = path.ccp_alphas

# Test different alphas
best_score = 0
best_alpha = 0

for alpha in alphas:
    dt_pruned = DecisionTreeClassifier(ccp_alpha=alpha)
    dt_pruned.fit(X_train, y_train)
    score = dt_pruned.score(X_val, y_val)
    
    if score > best_score:
        best_score = score
        best_alpha = alpha

# Final model with best alpha
dt_final = DecisionTreeClassifier(ccp_alpha=best_alpha)
dt_final.fit(X_train, y_train)
```

### Reduced Error Pruning

**Algorithm**:
1. Build full tree
2. For each node (bottom-up):
   - Remove the subtree
   - Check validation accuracy
   - Keep removal if accuracy improves
3. Repeat until no improvement

**Advantages**:
- Simple to understand
- Direct on validation data

### Minimal Error Pruning

**Algorithm**:
1. Build full tree
2. For each node:
   - Calculate error with subtree
   - Calculate error if converted to leaf
   - Choose option with lower error
3. Repeat recursively

## Hyperparameter Tuning

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
}

dt = DecisionTreeClassifier()

grid_search = GridSearchCV(
    dt, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'
)

grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")
```

### Random Search

For large parameter spaces, use random search instead of grid search.

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'max_depth': np.arange(2, 20),
    'min_samples_split': np.arange(2, 50),
    'min_samples_leaf': np.arange(1, 30),
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(),
    param_dist,
    n_iter=20,  # Test 20 combinations
    cv=5
)

random_search.fit(X, y)
```

## Cross-Validation for Better Estimates

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5)

# 5-fold cross-validation
scores = cross_val_score(
    dt, 
    X, 
    y, 
    cv=5,  # 5 folds
    scoring='accuracy'
)

print(f"Fold scores: {scores}")
print(f"Mean: {scores.mean()}")
print(f"Std: {scores.std()}")
```

### Stratified K-Fold (for imbalanced data)

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    DecisionTreeClassifier(max_depth=5),
    X,
    y,
    cv=skf,
    scoring='accuracy'
)
```

## Early Stopping

```python
from sklearn.tree import DecisionTreeClassifier

best_depth = 1
best_score = 0
previous_score = 0
no_improve_count = 0

for depth in range(1, 30):
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    
    score = dt.score(X_val, y_val)
    
    if score > best_score:
        best_score = score
        best_depth = depth
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    # Stop if no improvement for 5 iterations
    if no_improve_count >= 5:
        print(f"Early stopping at depth {depth}")
        break

print(f"Best depth: {best_depth} with score {best_score}")
```

## Ensemble Methods (Better Solution)

Instead of controlling a single tree, combine multiple trees:

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Multiple trees with random subsets reduce overfitting
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5
)

rf.fit(X_train, y_train)
```

**Benefits**:
- Voting reduces variance
- Better generalization
- More robust

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Iterative tree building with learning rate control
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

gb.fit(X_train, y_train)
```

## Monitoring Overfitting

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=5)

train_sizes, train_scores, val_scores = learning_curve(
    dt, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.legend()
plt.show()
```

**Interpretation**:
- Curves close together → Good generalization
- Large gap → Overfitting
- Both low → Underfitting

## Best Practices

1. **Use validation set** for hyperparameter tuning
2. **Cross-validate** for more robust estimates
3. **Start simple** (shallow trees) then increase complexity
4. **Monitor both** training and validation accuracy
5. **Use ensemble methods** for better performance
6. **Prune post-training** if overfitting detected
7. **Consider ensemble methods** for production models

## Summary

- Overfitting is common in decision trees
- Prevention through hyperparameter limits is easier than post-hoc pruning
- Validation set is essential for tuning
- Ensemble methods often provide better solutions
- Monitor learning curves to detect overfitting early
