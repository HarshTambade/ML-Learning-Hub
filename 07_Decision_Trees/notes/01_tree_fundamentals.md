# 01: Tree Fundamentals

## Core Concepts

### What is a Decision Tree?

A **decision tree** is a tree-structured classifier where:
- Each internal node represents a test on an attribute (feature)
- Each branch represents an outcome of the test
- Each leaf node represents a class label (for classification) or a value (for regression)
- The root node is the topmost decision node

### Basic Terminology

1. **Root Node**: The top node where the data is first split
2. **Internal/Decision Nodes**: Nodes that split the data based on feature values
3. **Leaf Nodes (Terminal Nodes)**: Final nodes that output predictions
4. **Branch**: A connection between nodes representing decision outcomes
5. **Depth**: The length of the longest path from root to leaf
6. **Height**: Maximum depth of the tree

## Tree Structure Example

```
                    Root: Feature A <= 5?
                           /          \
                         YES          NO
                        /              \
                   Feature B <= 3?    Predict Class 1
                   /          \
                 YES          NO
                /              \
         Predict Class 0   Predict Class 1
```

## How Decision Trees Work

### 1. Splitting Process
- Start with all samples in root
- For each node, find the best feature and split value
- Best split maximizes information gain or minimizes impurity
- Recursively repeat for child nodes
- Stop when reaching leaf (pure node), max depth, or min samples

### 2. Making Predictions
- Start at root node
- Follow the decision path based on feature values
- Reach a leaf node
- Output the prediction (class label or value)

## Decision Tree Types

### Classification Trees
- **Purpose**: Predict categorical outcomes
- **Output**: Class labels
- **Example**: Predicting if email is spam (Yes/No)
- **Common Algorithms**: ID3, C4.5, CART

### Regression Trees
- **Purpose**: Predict continuous values
- **Output**: Real-valued numbers
- **Example**: Predicting house prices
- **Leaf Values**: Usually mean of training samples in leaf

## Key Characteristics

### Advantages
1. **Interpretability**: Easy to visualize and understand
2. **No Feature Scaling**: Works with raw data
3. **Handles Non-linear**: Captures complex patterns
4. **Mixed Data**: Works with numeric AND categorical
5. **Fast Inference**: O(log n) prediction time
6. **Feature Importance**: Built-in importance scores

### Disadvantages
1. **Overfitting**: Can memorize training data
2. **Instability**: Small data changes cause big tree changes
3. **Class Imbalance**: Biased toward majority class
4. **High Variance**: Sensitive to training data variations
5. **Greedy**: Not globally optimal (local optimization)
6. **Small Perturbations**: Training data noise affects tree structure

## Mathematical Foundation

### Node Impurity

Measures how mixed the classes are at a node:

**Entropy** (Information Theory):
```
Entropy(S) = -sum(p_i * log2(p_i))
where p_i = proportion of samples of class i
```

**Gini Index** (CART):
```
Gini(S) = 1 - sum(p_i^2)
```

Both range from 0 (pure - all one class) to ~1 (impure - mixed classes)

## Tree Growing Criteria

### Stopping Conditions
1. **Pure Node**: All samples belong to one class
2. **Max Depth**: Reached maximum tree depth
3. **Min Samples Split**: Node has fewer samples than threshold
4. **Min Samples Leaf**: Would create leaf with fewer samples than threshold
5. **Max Leaf Nodes**: Reached maximum number of leaves
6. **Min Impurity Decrease**: Gain below threshold

## Practical Considerations

### When to Use Decision Trees
1. When interpretability is crucial
2. When data is mostly discrete/categorical
3. For quick baseline models
4. When feature engineering isn't feasible
5. For binary or multi-class problems

### When NOT to Use
1. When accuracy is paramount (use ensemble methods)
2. With highly imbalanced data
3. When features are continuous (XGBoost might be better)
4. With very small datasets (prone to overfitting)
5. When computational efficiency is critical
