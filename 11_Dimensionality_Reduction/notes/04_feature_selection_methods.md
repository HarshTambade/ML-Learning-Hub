# Feature Selection Methods

## Feature Selection Overview

Feature selection directly keeps original features, improving interpretability while reducing dimensionality.

## Categories

### 1. Univariate Methods
Evaluate each feature independently using statistical tests.

**SelectKBest**
- Selects k features with highest scores
- F-statistic for classification
- Mutual information for ranking

**Univariate Tests**
- Chi-square: categorical features
- ANOVA F-test: continuous features
- Mutual information: any type

### 2. Recursive Feature Elimination (RFE)
- Train model with all features
- Remove feature with lowest importance
- Repeat until k features remain
- Considers feature interactions

**Advantages**
- Accounts for feature interactions
- Works with any model
- Stable and reliable

**Disadvantages**
- Computationally expensive
- Slower than univariate methods

### 3. Model-Based Selection
Use model coefficients or feature importance.

**Tree-based Methods**
- Random Forest importance
- XGBoost feature importance
- Gradient Boosting importance

**Linear Models**
- Absolute value of coefficients
- Logistic regression weights

### 4. L1-based Selection
Use L1 regularization to shrink unimportant features to zero.

**Lasso (L1 Regression)**
- Built-in feature selection
- Continuous shrinkage
- Simple interpretation

## Comparison

| Method | Speed | Interactions | Interpretability |
|--------|-------|-------------|------------------|
| Univariate | Very Fast | No | High |
| RFE | Slow | Yes | High |
| Tree-based | Fast | Yes | Medium |
| L1 | Medium | No | High |

## Best Practices

1. **Use cross-validation** to validate k choice
2. **Compare multiple methods** - they may select different features
3. **Scale features** appropriately before selection
4. **Consider domain knowledge** - keep interpretable features
5. **Validate downstream** - check model performance
6. **Use ensemble** - combine selections from multiple methods

## When to Use Feature Selection

✓ Interpretability critical
✓ Limited number of features
✓ Want to reduce data collection
✓ Features have clear meaning

✗ Many redundant features
✗ Complex interactions important
✗ Very high dimensions (>10k)
