# Descriptive Statistics

## Overview
Descriptive statistics summarize and describe the main features of a dataset using statistical measures and visualizations.

## Measures of Central Tendency

### Mean (Average)
- Sum of all values divided by the number of values
- Formula: μ = (Σx) / n
- Sensitive to outliers
- Used for continuous data
- Example: [2, 4, 6, 8, 10] → Mean = 6

### Median
- Middle value when data is sorted
- Less sensitive to outliers
- Good for skewed distributions
- Example: [2, 4, 6, 8, 10] → Median = 6

### Mode
- Most frequently occurring value
- Useful for categorical data
- Can have multiple modes
- Example: [1, 2, 2, 3, 3, 3] → Mode = 3

## Measures of Dispersion

### Range
- Difference between maximum and minimum
- Formula: Range = Max - Min
- Simple but sensitive to outliers
- Example: [2, 4, 6, 8, 10] → Range = 10 - 2 = 8

### Variance
- Average squared deviation from mean
- Formula: σ² = Σ(x - μ)² / n
- Units are squared
- Useful for statistical calculations

### Standard Deviation
- Square root of variance
- Formula: σ = √Variance
- Same units as original data
- Indicates spread of data
- Rule of thumb (Normal Distribution):
  - 68% within 1 SD
  - 95% within 2 SD
  - 99.7% within 3 SD

### Interquartile Range (IQR)
- Difference between Q3 and Q1
- Formula: IQR = Q3 - Q1
- Robust to outliers
- Used for boxplots

## Shape of Distributions

### Skewness
- Measures asymmetry
- Formula: Skewness = E[(X - μ)³] / σ³
- Values:
  - Positive skew: tail on right (Mean > Median)
  - Negative skew: tail on left (Mean < Median)
  - Zero skew: symmetric

### Kurtosis
- Measures tail heaviness
- Formula: Kurtosis = E[(X - μ)⁴] / σ⁴
- Values:
  - Positive: heavier tails (leptokurtic)
  - Negative: lighter tails (platykurtic)
  - Zero: normal distribution (mesokurtic)

## Quantiles and Percentiles

### Quartiles
- Q1 (25th percentile): 25% of data below
- Q2 (50th percentile): Median
- Q3 (75th percentile): 75% of data below

### Percentiles
- nth percentile: n% of data is below this value
- Used to understand data distribution
- Example: 90th percentile in test scores

## Five-Number Summary
1. Minimum
2. Q1 (25th percentile)
3. Q2 (Median)
4. Q3 (75th percentile)
5. Maximum

Used to create boxplots for visualization.

## Association Between Variables

### Covariance
- Measures joint variability
- Formula: Cov(X,Y) = E[(X - μx)(Y - μy)]
- Positive: variables move together
- Negative: variables move opposite
- Units are product of original units

### Correlation
- Standardized covariance (-1 to +1)
- Formula: r = Cov(X,Y) / (σx × σy)
- -1: perfect negative
- 0: no linear relationship
- +1: perfect positive
- Unitless

## Common Distributions

### Normal Distribution
- Bell-shaped, symmetric
- Mean = Median = Mode
- Defined by mean and standard deviation
- Many natural phenomena

### Skewed Distributions
- Right-skewed: Mean > Median
- Left-skewed: Mean < Median
- Common in real-world data

## Outliers and Anomalies

### Detection Methods
1. **Z-score**: |Z| > 3 (approximately)
2. **IQR method**: Value < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
3. **Mahalanobis distance**: For multivariate data

### Handling Outliers
- **Keep**: If valid measurements
- **Transform**: Log or other transformations
- **Remove**: If data entry error
- **Separate analysis**: Study separately

## Statistical Tables and Displays

### Frequency Distribution
- Count of occurrences per class
- Useful for understanding data patterns

### Histogram
- Visual of frequency distribution
- Shows shape and spread

### Boxplot
- Shows five-number summary
- Easy outlier identification

### Scatter Plot
- Shows relationship between two variables
- Helps identify patterns

## Summary Statistics Examples

### Dataset: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
- Mean: 11
- Median: 11
- Mode: No mode (all values once)
- Range: 18
- Variance: 33
- Standard Deviation: 5.74
- Q1: 6.25
- Q3: 15.75
- IQR: 9.5

## Applications in Machine Learning

1. **Feature Understanding**: Explore data distributions
2. **Data Quality Check**: Identify anomalies
3. **Normalization**: Transform features (z-score, min-max)
4. **Feature Selection**: Correlation analysis
5. **Model Interpretation**: Understanding predictions

## Key Takeaways

✓ Descriptive statistics summarize data characteristics
✓ Choose appropriate measures based on data type
✓ Always visualize data before analysis
✓ Be aware of outliers and their impact
✓ Understand the shape of distributions
✓ Correlation ≠ Causation

## Common Pitfalls

❌ Using mean for skewed data (use median)
❌ Ignoring outliers completely
❌ Assuming normal distribution without testing
❌ Confusing correlation with causation
❌ Using inappropriate visualizations

## Further Reading
- Chapter 1 in "Introduction to Statistical Learning"
- Khan Academy: Descriptive Statistics
- Visualizations: matplotlib, seaborn tutorials
