# 03: Entropy Calculation - Step by Step

## What is Entropy?

Entropy measures the **uncertainty or disorder** in a dataset. In information theory, it quantifies the average information (or surprise) when we randomly select a sample from the dataset.

## The Formula

```
Entropy(S) = -sum(p_i * log2(p_i))

where:
- S = dataset (or node)
- p_i = proportion/probability of class i
- log2 = logarithm base 2
- sum = over all classes present
```

### Why Base 2?
- Log base 2 gives entropy in **bits**
- Aligns with information theory concepts
- Easy to interpret: max entropy for binary is 1 bit

## Understanding the Formula

### Component Breakdown

1. **p_i**: Probability of class i
   - Count samples of class i / Total samples

2. **log2(p_i)**: Information content
   - Higher probability = lower information value
   - log2(1) = 0 (no surprise if guaranteed)
   - log2(0.5) = -1 (maximum surprise for binary)

3. **p_i * log2(p_i)**: Weighted information
   - More frequent classes contribute less to entropy

4. **Negative sign**: Convention
   - Makes result positive
   - log2(p) is negative when p < 1

## Key Values to Remember

```
log2(1.00) = 0.000
log2(0.75) ≈ -0.415
log2(0.67) ≈ -0.585
log2(0.50) = -1.000
log2(0.33) ≈ -1.585
log2(0.25) = -2.000
```

## Worked Examples

### Example 1: Pure Dataset (All One Class)

**Dataset**: 10 samples, all Class A

```
Class Distribution:
- Class A: 10/10 = 1.0

Entropy(S) = -(1.0 * log2(1.0))
           = -(1.0 * 0)
           = 0
```

**Interpretation**: No uncertainty. We always know it's Class A.

### Example 2: Balanced Binary Classification

**Dataset**: 10 samples
- Class A: 5 samples
- Class B: 5 samples

```
Class Probabilities:
- p_A = 5/10 = 0.5
- p_B = 5/10 = 0.5

Entropy(S) = -(0.5 * log2(0.5)) - (0.5 * log2(0.5))
           = -(0.5 * (-1)) - (0.5 * (-1))
           = 0.5 + 0.5
           = 1.0 bit
```

**Interpretation**: Maximum uncertainty for binary. Equal chance of either class.

### Example 3: Imbalanced Binary Classification

**Dataset**: 10 samples
- Class A: 7 samples
- Class B: 3 samples

```
Class Probabilities:
- p_A = 7/10 = 0.7
- p_B = 3/10 = 0.3

Log Values:
- log2(0.7) ≈ -0.515
- log2(0.3) ≈ -1.737

Entropy(S) = -(0.7 * (-0.515)) - (0.3 * (-1.737))
           = 0.361 + 0.521
           = 0.882 bits
```

**Interpretation**: Less uncertain than balanced, but still significant disorder.

### Example 4: Three-Class Problem

**Dataset**: 12 samples
- Class A: 6 samples
- Class B: 4 samples
- Class C: 2 samples

```
Class Probabilities:
- p_A = 6/12 = 0.5
- p_B = 4/12 ≈ 0.333
- p_C = 2/12 ≈ 0.167

Log Values:
- log2(0.5) = -1.0
- log2(0.333) ≈ -1.585
- log2(0.167) ≈ -2.585

Entropy(S) = -(0.5 * (-1.0)) - (0.333 * (-1.585)) - (0.167 * (-2.585))
           = 0.5 + 0.528 + 0.432
           = 1.460 bits
```

**Interpretation**: More uncertainty than binary with same dominant class.

## Entropy Range

For a dataset with k classes:
- **Minimum**: 0 (when all samples are one class - pure)
- **Maximum**: log2(k) (when all classes equally distributed)

### Examples:
- Binary classification: [0, 1]
- Three classes: [0, 1.585]
- Four classes: [0, 2]
- Ten classes: [0, 3.322]

## Information Gain Calculation

Now that we understand entropy, we can calculate information gain for splits:

```
IG(S, A) = Entropy(S) - sum(|S_v|/|S| * Entropy(S_v))
```

### Example: Calculating Information Gain

**Parent Node**:
- 10 samples: 7 A, 3 B
- Entropy = 0.882 (from Example 3 above)

**Split on Feature X**:
- Left branch: 6 samples (5 A, 1 B)
  ```
  Entropy_left = -(5/6 * log2(5/6)) - (1/6 * log2(1/6))
               ≈ -(0.833 * (-0.263)) - (0.167 * (-2.585))
               ≈ 0.219 + 0.431
               ≈ 0.650
  ```

- Right branch: 4 samples (2 A, 2 B)
  ```
  Entropy_right = -(2/4 * log2(2/4)) - (2/4 * log2(2/4))
                = -(0.5 * (-1)) - (0.5 * (-1))
                = 0.5 + 0.5
                = 1.0
  ```

**Weighted Average**:
```
Weighted_entropy = (6/10 * 0.650) + (4/10 * 1.0)
                 = 0.390 + 0.400
                 = 0.790
```

**Information Gain**:
```
IG = 0.882 - 0.790 = 0.092 bits
```

**Interpretation**: This split reduces entropy by 0.092 bits.

## Common Calculation Mistakes

### Mistake 1: Forgetting the Negative Sign
```
WRONG: Entropy = sum(p_i * log2(p_i))
RIGHT: Entropy = -sum(p_i * log2(p_i))
```

### Mistake 2: Using Natural Log Instead of Log2
```
WRONG: Entropy = -sum(p_i * ln(p_i))
RIGHT: Entropy = -sum(p_i * log2(p_i))
```
(Natural log gives nats instead of bits)

### Mistake 3: Not Handling Zero Probability
```
Problem: log2(0) is undefined!
Solution: If p_i = 0, then p_i * log2(p_i) = 0
(Use limit: lim(p->0) p*log(p) = 0)
```

### Mistake 4: Wrong Weighted Average
```
WRONG: (|S_left| + |S_right|) / 2 * entropy
RIGHT: (|S_left|/|S| * entropy_left) + (|S_right|/|S| * entropy_right)
```

## Practical Implementation

### Python Calculation

```python
import math

def entropy(class_counts):
    """Calculate entropy given class counts"""
    total = sum(class_counts)
    ent = 0.0
    
    for count in class_counts:
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    
    return ent

# Example
classes = [7, 3]  # 7 of class A, 3 of class B
print(entropy(classes))  # Output: 0.882...
```

## Summary

- Entropy = measure of uncertainty/disorder
- Higher entropy = more mixed classes
- Lower entropy = more pure (homogeneous)
- Information Gain = reduction in entropy after split
- Base 2 logarithm gives results in bits
- Key for building optimal decision trees
