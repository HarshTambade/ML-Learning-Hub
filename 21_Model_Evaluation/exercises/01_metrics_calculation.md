
# Exercise 1: Classification Metrics Calculation - Detailed Guide

## Objective
Understand, calculate, and interpret classification metrics from scratch. This exercise teaches the fundamentals of model evaluation in classification tasks through hands-on calculation and visualization.

## Background Theory

### Confusion Matrix Basics
A confusion matrix is a table used to describe the performance of a classification model. It contains:
- **True Positives (TP)**: Correctly predicted positive cases
- **True Negatives (TN)**: Correctly predicted negative cases
- **False Positives (FP)**: Negative cases incorrectly predicted as positive (Type I error)
- **False Negatives (FN)**: Positive cases incorrectly predicted as negative (Type II error)

### Dataset Information
```
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]
```

### Building the Confusion Matrix
For each prediction:
- Index 0: True=0, Pred=0 → TN ✓
- Index 1: True=1, Pred=1 → TP ✓
- Index 2: True=1, Pred=0 → FN ✗
- Index 3: True=0, Pred=0 → TN ✓
- Index 4: True=1, Pred=1 → TP ✓
- Index 5: True=0, Pred=1 → FP ✗
- Index 6: True=1, Pred=1 → TP ✓
- Index 7: True=1, Pred=1 → TP ✓

**Confusion Matrix:**
```
        Predicted=0  Predicted=1
True=0      2            1        (TN=2, FP=1)
True=1      1            4        (FN=1, TP=4)
```

## Task 1: Manual Metric Calculations

### Step 1: Calculate Accuracy
**Formula:** Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Calculation:**
- TP = 4, TN = 2, FP = 1, FN = 1
- Accuracy = (4 + 2) / (4 + 2 + 1 + 1) = 6/8 = **0.75 or 75%**

**Interpretation:** The model correctly classifies 75% of all samples.

### Step 2: Calculate Precision
**Formula:** Precision = TP / (TP + FP)

**Calculation:**
- Precision = 4 / (4 + 1) = 4/5 = **0.80 or 80%**

**Interpretation:** When the model predicts positive, it is correct 80% of the time. This metric focuses on reducing false positives (useful when false positives are costly).

### Step 3: Calculate Recall (Sensitivity/True Positive Rate)
**Formula:** Recall = TP / (TP + FN)

**Calculation:**
- Recall = 4 / (4 + 1) = 4/5 = **0.80 or 80%**

**Interpretation:** Of all actual positive cases, the model correctly identifies 80%. This metric is important when false negatives are costly (e.g., disease detection).

### Step 4: Calculate F1-Score
**Formula:** F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

**Calculation:**
- F1-Score = 2 × (0.80 × 0.80) / (0.80 + 0.80) = 2 × 0.64 / 1.60 = 1.28 / 1.60 = **0.80 or 80%**

**Interpretation:** The harmonic mean of precision and recall. Useful when you need balance between both metrics.

### Step 5: Additional Metrics

**Specificity (True Negative Rate):**
- Formula: Specificity = TN / (TN + FP) = 2 / (2 + 1) = 2/3 = **0.667 or 66.7%**

**False Positive Rate (FPR):**
- Formula: FPR = FP / (FP + TN) = 1 / (1 + 2) = 1/3 = **0.333 or 33.3%**

**False Negative Rate (FNR):**
- Formula: FNR = FN / (FN + TP) = 1 / (1 + 4) = 1/5 = **0.20 or 20%**

## Task 2: Verification with Scikit-Learn

```python
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)

y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Individual Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
```

## Task 3: Interpretation & Analysis

### What do the metrics tell us?

1. **Accuracy (75%):** While this seems decent, accuracy alone can be misleading in imbalanced datasets.

2. **Precision vs Recall Trade-off:**
   - Both are 80%, suggesting balanced performance
   - No clear preference for positive over negative cases
   - One false positive and one false negative

3. **When to prioritize which metric?**
   - **High Precision needed:** Cancer detection (minimize false alarms)
   - **High Recall needed:** Disease screening (catch all cases)
   - **Balanced:** Email spam detection

## Task 4: Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Classification Metrics Comparison')
plt.ylim([0, 1])
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.show()
```

## Task 5: Summary Table

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| Accuracy | (TP+TN)/(Total) | 0.75 | Overall correctness |
| Precision | TP/(TP+FP) | 0.80 | Positive prediction accuracy |
| Recall | TP/(TP+FN) | 0.80 | Positive case detection rate |
| F1-Score | 2×(P×R)/(P+R) | 0.80 | Harmonic mean |
| Specificity | TN/(TN+FP) | 0.67 | Negative case detection rate |

## Learning Outcomes

After completing this exercise, you will:
1. ✓ Understand the confusion matrix concept
2. ✓ Calculate metrics from scratch
3. ✓ Interpret each metric's meaning
4. ✓ Know when to use each metric
5. ✓ Implement using scikit-learn
6. ✓ Visualize results effectively

## Key Takeaways

- **Accuracy alone is insufficient** - use multiple metrics
- **Context matters** - choose metrics based on business requirements
- **Trade-offs exist** - precision vs recall is a key consideration
- **Confusion matrix is fundamental** - understand it deeply
- **Automation helps** - scikit-learn verification confirms manual calculations

## Further Challenges

1. Try with different prediction sets
2. Explore class imbalance effects
3. Implement ROC curve
4. Calculate AUC score
5. Multi-class classification metrics
