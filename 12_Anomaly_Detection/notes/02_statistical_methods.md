# Statistical Methods for Anomaly Detection

## Overview

Statistical methods are classical approaches that assume data follows known distributions. They're fast, interpretable, and work well for univariate data.

## Z-Score Method

**Formula**: `Z = (X - μ) / σ`

**Principle**: Points with |Z| > 3 are considered anomalies (3-sigma rule)

**Advantages**:
- Simple and fast
- Interpretable
- No hyperparameters

**Disadvantages**:
- Assumes normal distribution
- Sensitive to extreme values
- Not robust

## Modified Z-Score (MAD)

Uses Median Absolute Deviation instead of standard deviation for robustness.

**Formula**: `Modified Z = 0.6745 * (X - median) / MAD`

**Threshold**: Typically 3.5

**Better than Z-Score**: Less sensitive to outliers

## Interquartile Range (IQR)

**Process**:
1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
2. IQR = Q3 - Q1
3. Lower bound = Q1 - 1.5 * IQR
4. Upper bound = Q3 + 1.5 * IQR
5. Points outside bounds are anomalies

**Advantages**:
- Distribution-free
- Robust to outliers
- Easy to understand

## Mahalanobis Distance

**For Multivariate Data**:

`D = sqrt((X - μ)ᵀ Σ⁻¹ (X - μ))`

Where:
- μ: Mean vector
- Σ: Covariance matrix
- Σ⁻¹: Inverse of covariance

**Advantages**:
- Accounts for correlations
- Works in high dimensions
- Considers variable relationships

**Disadvantages**:
- Requires covariance estimation
- Computationally expensive
- Sensitive to singular covariance

## Comparison

| Method | Type | Assumptions | Multivariate | Speed |
|--------|------|-------------|--------------|-------|
| Z-Score | Parametric | Normal | ❌ | ⭐⭐⭐ |
| Modified Z | Parametric | Normal (robust) | ❌ | ⭐⭐⭐ |
| IQR | Non-parametric | None | ❌ | ⭐⭐⭐ |
| Mahalanobis | Parametric | Normal (multivariate) | ✅ | ⭐⭐ |

## When to Use

- **Z-Score**: Univariate, normally distributed data
- **Modified Z**: Univariate, robust needed
- **IQR**: Any distribution, simple cases
- **Mahalanobis**: Multivariate, correlated features

## Implementation Considerations

1. **Standardize data** before applying
2. **Choose appropriate threshold** based on domain
3. **Validate assumptions** (especially normality)
4. **Monitor false positives** in production
5. **Update thresholds** as data evolves

---
*Next: Learn about Isolation Forest*
