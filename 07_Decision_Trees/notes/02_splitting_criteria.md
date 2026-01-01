# 02: Splitting Criteria

## Introduction to Splitting

The core of decision tree learning is finding the **best split** at each node. This is where impurity measures come into play. They help us determine which feature and threshold value will best separate the classes.

## Gini Impurity (CART)

### Definition

Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled.

```
Gini(S) = 1 - sum(p_i^2)

where:
- S = dataset at node
- p_i = proportion of class i in S
- sum is over all classes
```

### Range
- **Gini = 0**: Pure node (all samples from one class)
- **Gini = 0.5**: For binary classification, maximum impurity (50-50 split)
- **Gini = 1**: Worst case (all classes equally distributed)

### Example: Gini Calculation

**Dataset**: 10 samples
- Class A: 6 samples
- Class B: 4 samples

```
Gini(S) = 1 - (6/10)^2 - (4/10)^2
        = 1 - 0.36 - 0.16
        = 1 - 0.52
        = 0.48
```

### Gini Gain

Measures the reduction in impurity after a split:

```
Gain_Gini(S, A) = Gini(S) - sum(|S_v|/|S| * Gini(S_v))

where:
- S_v = subset of S after split on attribute A with value v
```

**Higher Gini Gain = Better Split**

## Entropy (ID3, C4.5)

### Definition

Entropy measures the amount of disorder or uncertainty in a dataset.

```
Entropy(S) = -sum(p_i * log2(p_i))

where:
- p_i = proportion of class i
- log2 = logarithm base 2
- sum is over all classes
```

### Range
- **Entropy = 0**: Pure node (one class only)
- **Entropy = 1**: Maximum uncertainty (equal classes in binary)
- **Entropy = log2(k)**: k classes, all equally distributed

### Example: Entropy Calculation

**Dataset**: 10 samples
- Class A: 6 samples (p_A = 0.6)
- Class B: 4 samples (p_B = 0.4)

```
Entropy(S) = -(0.6 * log2(0.6)) - (0.4 * log2(0.4))
           = -(0.6 * -0.737) - (0.4 * -1.322)
           = 0.442 + 0.529
           = 0.971
```

## Information Gain

### Definition

Information Gain measures how much entropy decreases after a split.

```
IG(S, A) = Entropy(S) - sum(|S_v|/|S| * Entropy(S_v))

where:
- S = parent node
- A = attribute (feature)
- S_v = subset of samples with value v for attribute A
```

### Example: Information Gain

**Parent Node**:
- Total: 10 samples
- Class A: 6, Class B: 4
- Entropy(S) = 0.971

**Split on Feature X**:
- Left child: 5 samples (3 A, 2 B) → Entropy = 0.971
- Right child: 5 samples (3 A, 2 B) → Entropy = 0.971

```
IG = 0.971 - (5/10 * 0.971) - (5/10 * 0.971)
   = 0.971 - 0.971
   = 0
```

This split provides no information gain!

**Better Split on Feature Y**:
- Left child: 4 samples (4 A, 0 B) → Entropy = 0
- Right child: 6 samples (2 A, 4 B) → Entropy = 0.918

```
IG = 0.971 - (4/10 * 0) - (6/10 * 0.918)
   = 0.971 - 0.551
   = 0.420
```

This is a much better split!

## Comparing Gini vs Entropy

| Aspect | Gini | Entropy |
|--------|------|----------|
| **Formula** | 1 - sum(p_i^2) | -sum(p_i*log2(p_i)) |
| **Range** | [0, 0.5] for binary | [0, 1] for binary |
| **Computation** | Faster (no log) | Slower (log calculation) |
| **Algorithm** | CART | ID3, C4.5 |
| **Interpretation** | Misclassification probability | Information disorder |
| **Bias** | Favors features with many values | Can be biased toward features with many splits |
| **Performance** | Generally similar results | Generally similar results |

### In Practice
- **Similar Performance**: Results are usually very similar
- **Gini Faster**: No logarithm calculation
- **Entropy Theory**: More information-theoretic foundation
- **Choose Based On**: Computational constraints or convention

## Split Selection Algorithm

### For Each Node:

1. **For each feature**:
   - Sort unique values
   - For each possible threshold:
     - Calculate resulting split quality
   - Find best threshold for this feature

2. **Compare all features**:
   - Select feature with highest gain/lowest impurity

3. **Create split**:
   - Left child: samples <= threshold
   - Right child: samples > threshold

## Handling Categorical Features

### Binary Split (CART)
- Even for categorical: splits into two groups
- Finds best grouping of categories

### Multi-way Split (ID3, C4.5)
- Can split on each category value
- Creates multiple branches from single feature

## Gain Ratio (C4.5)

Addresses bias toward features with many values:

```
Gain_Ratio(S, A) = IG(S, A) / SplitInfo(S, A)

where:
SplitInfo(S, A) = -sum(|S_v|/|S| * log2(|S_v|/|S|))
```

**Effect**: Normalizes information gain by split complexity

## Chi-Square Test

Statistical approach to test split significance:

```
chi2 = sum((Observed - Expected)^2 / Expected)
```

**Usage**: Can use p-value to decide if split is significant

## Summary

- **Gini Impurity**: Probability-based, faster computation
- **Entropy/Info Gain**: Information-theoretic foundation
- **Both**: Generally produce similar trees
- **Choice**: Based on algorithm convention or efficiency needs
- **Key Point**: Greedy selection finds locally optimal splits
