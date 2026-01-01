# 04: Tree Visualization and Structure

## Understanding Tree Structure

Visualizing decision trees helps us understand how the algorithm makes decisions. Each node and branch tells a story about feature importance and decision boundaries.

## Basic Tree Diagram

### Notation

```
     [Root Node]
       /        \
     /          \
  [Branch]    [Branch]
   /  \        /   \
  /    \      /     \
 [Node][Node][Node][Leaf]
```

### Simple Example: Iris Flower Classification

```
            Petal Width <= 0.8?
                 /  \
              Yes    No
              /        \
        [Class: setosa]  Sepal Length <= 6.9?
                          /         \
                       Yes           No
                      /               \
           [Class: versicolor]  [Class: virginica]
```

## Tree with Detailed Information

### What to Include in Nodes

```
┌─────────────────────────────┐
│  Feature <= Threshold?      │  (Decision Rule)
├─────────────────────────────┤
│  samples = 150              │  (Total samples at node)
├─────────────────────────────┤
│  value = [50, 50, 50]       │  (Class distribution)
├─────────────────────────────┤
│  Class: setosa              │  (Predicted class)
│  Gini = 0.667 or 1.0        │  (Impurity measure)
└─────────────────────────────┘
```

## Decision Boundaries Visualization

### 2D Example: Iris Dataset

```
  7 |                 X
    |            X    V  V
    | V   V  V        V   V
  6 |V V V        |
    |       V     |
  5 |    V  |  V  |  O
    |  ------+-----|---  Sepal Length = 6.0
  4 | S      |  O  O  O
    | S  S   |        
  3 | S      |        
    |________|________
    2    3    4    5    6
       Sepal Width
      
    S = Setosa
    V = Versicolor
    O = Virginica
    | = Decision boundary
```

## Binary vs Multi-way Splits

### Binary Split (CART)

Every split creates exactly 2 branches:

```
         Feature X <= 5
          /        \
       YES          NO
       /             \
   [Feature Y <= 3] [Class A]
    /        \
  YES        NO
  /           \
[Class B]  [Class C]
```

### Multi-way Split (ID3, C4.5)

A single feature can create multiple branches:

```
         Color?
    /     |     |     \
  Red   Blue  Green  Yellow
  /       |      |      \
 [A]    [B]    [B]     [C]
```

## Tree Depth and Complexity

### Shallow Tree (max_depth=2)

```
Depth 0:            [Root]
                    /    \
                   /      \
Depth 1:      [Node]    [Node]
              /   \      /   \
Depth 2:     [L] [L]   [L]  [L]

+ Underfitting (high bias, low variance)
+ Fast predictions
- May miss patterns
```

### Deep Tree (max_depth=5)

```
                    [Root]
            /       /     \      \
           /       /       \      \
Depth 1:  [N]    [N]      [N]    [N]
          |\      |\       |\     |
          | \     | \      | \    |
         [N][N] [N][N]   [N][N] [N][N]
          |    |    |  |   |   |   |
         [L][L][L][L][L][L][L][L][L]

- Overfitting (low bias, high variance)
- Slow predictions
+ Captures patterns
```

## Visualizing Feature Importance

### Feature Importance from Tree

```
Feature        Importance
─────────────────────────
Petal Width    0.42   ███████████
Petal Length   0.28   ████████
Sepal Length   0.20   ██████
Sepal Width    0.10   ███

Higher bar = more important feature
```

### How It's Calculated

Importance at each split:
```
Importance = (samples_at_split / total_samples) * Gain
```

## Tree Metrics Visualization

### Sample Distribution

```
At node with 100 samples:

[████████████████████] 100 samples

After split:

Left:  [███████████] 65 samples
Right: [█████] 35 samples
```

### Impurity Decrease

```
Parent Gini: 0.667  [████████]
             
Left Gini:   0.375  [█████]
Right Gini:  0.500  [██████]

Weighted:    0.462  [██████]
Gain:        0.205  [██]
```

## Visual Interpretation Tips

### Node Color Intensity

Darker = Purer (more homogeneous class distribution)

```
Pure class distribution:
[████████████████] 100% one class  (Black)

Mixed distribution:
[████    ████    ████    ████]      (Gray)
```

### Text Size in Nodes

Larger text = More important features or larger nodes

```
Large text: This feature splits many samples
Small text: This feature splits few samples
```

## Tree Structure Patterns

### Balanced Tree

```
        |Root|
       /     \
    |N|       |N|
   /  \     /  \
  |L| |L| |L| |L|

Characteristic: Even distribution
✓ Better generalization
```

### Left-Skewed Tree

```
        |Root|
       /     \
    |N|      |L|
   /  \
  |L| |N|
      / \
    |L| |L|

Characteristic: Complex left side
✗ May indicate outliers on one side
```

### Right-Skewed Tree

```
        |Root|
       /     \
    |L|      |N|
           /   \
         |L|   |N|
              /  \
            |L|  |L|

Characteristic: Linear decision boundary on right
✓ Clear separability
```

## Reading Decision Paths

### Following a Path to Prediction

**Example: Predicting for new sample**

Features: (Sepal_Length=5.5, Sepal_Width=3.2, ...)

```
Start: [Root: Petal Width <= 0.8?]
       ↓ No (Petal Width = 1.2)
       
[Node: Sepal Length <= 6.9?]
       ↓ No (Sepal Length = 5.5)
       
[Node: Petal Length <= 4.95?]
       ↓ Yes (Petal Length = 4.5)
       
[Leaf: CLASS = Versicolor]
```

## Practical Visualization Tools

### Python Visualization

```python
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

iris = load_iris()
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(iris.data, iris.target)

# Method 1: plot_tree (scikit-learn >= 0.21)
plt.figure(figsize=(12, 8))
tree.plot_tree(dt, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()

# Method 2: export_graphviz + graphviz
from sklearn.tree import export_graphviz
export_graphviz(dt, out_file='tree.dot')
# Then: dot -Tpng tree.dot -o tree.png
```

## Key Visualization Insights

1. **Root Node**: Most important feature and largest samples
2. **Leaf Nodes**: Pure or nearly pure classes (preferred)
3. **Branch Length**: Feature separation strength
4. **Tree Width**: Number of leaf nodes (complexity)
5. **Tree Symmetry**: Indicates balanced decision making

## Summary

- Visualize root → leaf paths to understand decisions
- Node colors/sizes indicate purity and importance
- Deep balanced trees risk overfitting
- Feature importance shows decision contributions
- Visualization helps identify patterns and anomalies
