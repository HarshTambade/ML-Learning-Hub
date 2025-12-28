# Chapter 03 Exercises: Statistics & Probability

## Basic Exercises

### 1. Descriptive Statistics

**Problem**: You have a dataset of 10 test scores: [65, 72, 68, 75, 70, 82, 78, 80, 76, 88]

- Calculate mean, median, and mode
- Find variance and standard deviation
- Identify Q1, Q2 (median), and Q3
- Calculate IQR and identify any outliers
- Determine skewness

**Solution Guide**:
```python
import numpy as np
from scipy import stats

scores = [65, 72, 68, 75, 70, 82, 78, 80, 76, 88]
mean = np.mean(scores)  # 75.4
median = np.median(scores)  # 77
q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
iqr = q3 - q1
outlier_low = q1 - 1.5 * iqr
outlier_high = q3 + 1.5 * iqr
```

### 2. Probability Basics

**Problem**: A deck of 52 playing cards. Find the probability of:
- Drawing a red card
- Drawing an Ace
- Drawing a red Ace
- Drawing a card that is red OR an Ace

**Solution**: P(Red) = 26/52, P(Ace) = 4/52, P(Red AND Ace) = 2/52, P(Red OR Ace) = 28/52

### 3. Normal Distribution

**Problem**: Heights of adult males are normally distributed with mean 175cm and std 10cm.
- What percentage are between 165-185cm?
- What height represents the 90th percentile?
- What percentage are taller than 200cm?

**Solution Guide**:
```python
from scipy.stats import norm
mean, std = 175, 10
p_165_185 = norm.cdf(185, mean, std) - norm.cdf(165, mean, std)  # 0.6827
percentile_90 = norm.ppf(0.90, mean, std)  # ~187.8cm
p_taller_200 = 1 - norm.cdf(200, mean, std)  # ~0.0062
```

### 4. Correlation

**Problem**: Calculate correlation between:
- X = [1, 2, 3, 4, 5]
- Y = [2, 4, 5, 4, 6]

Interpret the relationship and explain why it's not perfect.

**Solution**: Use `scipy.stats.pearsonr()` for Pearson correlation coefficient and p-value.

## Intermediate Exercises

### 5. Hypothesis Testing - t-test

**Problem**: A coffee shop claims their average customer spends 30 minutes. You collect 25 samples:
[25, 28, 32, 29, 31, 26, 33, 27, 30, 35, 24, 28, 31, 29, 32, 26, 28, 30, 33, 27, 29, 31, 32, 28, 30]

- Set up H0 and H1
- Perform a two-tailed t-test
- Interpret results at α=0.05

**Solution**: Use `scipy.stats.ttest_1samp()` to test against μ=30

### 6. Chi-square Test

**Problem**: Test if a dice is fair (expected 100 rolls per face).
Observed: [95, 108, 102, 97, 106, 92]

- Perform chi-square goodness-of-fit test
- Determine if dice is fair at α=0.05

### 7. ANOVA

**Problem**: Compare test scores across 3 teaching methods
- Method A: [72, 75, 78, 80, 76]
- Method B: [68, 70, 72, 69, 71]
- Method C: [85, 88, 82, 86, 89]

Test if there's significant difference using one-way ANOVA.

### 8. Confidence Intervals

**Problem**: Sample of 30 product weights with mean=100g, std=5g
- Calculate 95% confidence interval
- Calculate 99% confidence interval
- Interpret the difference

> **💡 Intermediate Exercise Reference**: For detailed examples on advanced hypothesis testing techniques including ANOVA, Levene's test, Shapiro-Wilk test, and effect size calculations, see [`code_examples/02_hypothesis_testing_advanced.py`](../code_examples/02_hypothesis_testing_advanced.py)

## Advanced Exercises

### 9. Linear Regression Analysis

**Problem**: Analyze relationship between study hours and exam scores:
```
Study Hours: [2, 3, 4, 5, 6, 7, 8]
Exam Scores: [55, 60, 65, 70, 75, 85, 90]
```

- Calculate correlation coefficient
- Fit linear regression model
- Interpret slope and intercept
- Calculate R-squared
- Make prediction for 6.5 hours

### 10. Multiple Comparisons Problem

**Problem**: You run 20 statistical tests at α=0.05 each.
- What's the probability of at least one Type I error (family-wise error rate)?
- How would you correct for this using Bonferroni correction?
- Apply correction and explain new α value

### 11. Power Analysis

**Problem**: Design study to detect effect size d=0.5 between two groups
- Using α=0.05 (two-tailed) and desired power=0.80
- Calculate required sample size per group
- How does sample size change if you want power=0.90?

### 12. Bayesian Probability

**Problem**: Medical test accuracy:
- Disease prevalence: 1% of population
- Test sensitivity: 95% (true positive rate)
- Test specificity: 90% (true negative rate)
- If test is positive, what's probability person has disease?

**Solution**: Use Bayes' theorem to calculate posterior probability

## Projects

### Project A: EDA and Statistical Summary
Choose a dataset from Kaggle and:
1. Calculate comprehensive descriptive statistics
2. Identify distributions for each variable
3. Test for outliers
4. Calculate correlations between variables
5. Create visualization summary

### Project B: A/B Testing Analysis
Design and analyze an A/B test:
1. Define hypothesis
2. Determine sample size needed
3. Analyze results using appropriate test
4. Calculate effect size
5. Report findings with confidence intervals

### Project C: Hypothesis Testing on Real Data
Perform comprehensive hypothesis tests on a real dataset:
1. State research questions
2. Choose appropriate tests
3. Verify assumptions
4. Report with p-values and confidence intervals
5. Discuss limitations

## Answer Key Structure

For each problem, your answer should include:
1. **Problem Understanding**: State what you're solving
2. **Method**: Explain statistical technique used
3. **Assumptions**: List and verify assumptions
4. **Calculations**: Show work or code
5. **Interpretation**: Explain what results mean
6. **Conclusion**: Final answer with context

## Common Pitfalls to Avoid

❌ Assuming normality without testing
❌ Ignoring sample size effects
❌ Confusing correlation with causation
❌ Multiple comparisons without correction
❌ Reporting only p-values without effect size
❌ Using inappropriate tests for data type
❌ Violating test assumptions

## Resources for Help

- Khan Academy: Statistics & Probability
- StatQuest with Josh Starmer: Detailed explanations
- SciPy Documentation: Statistical functions
- Your textbook: Worked examples

## Submission Checklist

- [ ] All exercises attempted
- [ ] Code properly commented
- [ ] Assumptions checked
- [ ] Results interpreted correctly
- [ ] Visualizations included where appropriate
- [ ] Work is reproducible
