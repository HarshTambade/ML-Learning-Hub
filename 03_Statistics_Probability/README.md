# Chapter 03: Statistics & Probability

Comprehensive guide to statistical methods and probability theory essential for Machine Learning.

## Overview
This chapter covers fundamental statistical concepts and probability theory that form the foundation for machine learning. Understanding these concepts is crucial for data analysis, model building, and interpretation.

## Learning Objectives
By the end of this chapter, you will understand:
- Descriptive statistics and data summarization
- Probability fundamentals and distributions
- Hypothesis testing and statistical inference
- Correlation and covariance analysis
- Bayesian thinking and applications
- Central Limit Theorem
- Statistical power and effect size

## Topics Covered

### 1. Descriptive Statistics
- **Measures of Central Tendency**: Mean, Median, Mode
- **Measures of Dispersion**: Variance, Standard Deviation, Range, IQR
- **Skewness and Kurtosis**: Shape of distributions
- **Summary Statistics**: Quantiles, Percentiles

### 2. Probability Fundamentals
- **Basic Concepts**: Events, Sample Space, Probability Axioms
- **Conditional Probability**: P(A|B), Independence
- **Bayes' Theorem**: Prior, Likelihood, Posterior
- **Counting Methods**: Permutations, Combinations

### 3. Probability Distributions
- **Discrete Distributions**:
  - Binomial Distribution
  - Poisson Distribution
  - Geometric Distribution
  - Hypergeometric Distribution

- **Continuous Distributions**:
  - Normal (Gaussian) Distribution
  - Exponential Distribution
  - Uniform Distribution
  - Beta Distribution
  - Gamma Distribution
  - Chi-square Distribution
  - t-distribution
  - F-distribution

### 4. Sampling and Estimation
- **Sampling Distributions**
- **Point Estimation**: Bias, Consistency, Efficiency
- **Confidence Intervals**: Construction and Interpretation
- **Maximum Likelihood Estimation (MLE)**

### 5. Hypothesis Testing
- **Null and Alternative Hypotheses**
- **Type I and Type II Errors**: Alpha and Beta
- **Test Statistics and p-values**
- **Parametric Tests**: t-test, z-test, chi-square test
- **Non-parametric Tests**: Mann-Whitney U, Kruskal-Wallis
- **Multiple Testing Correction**: Bonferroni, FDR

### 6. Analysis of Variance (ANOVA)
- **One-way ANOVA**
- **Two-way ANOVA**
- **Post-hoc Tests**: Tukey, Scheffe

### 7. Correlation and Regression
- **Pearson Correlation Coefficient**
- **Spearman Rank Correlation**
- **Covariance**
- **Simple Linear Regression**
- **Multiple Linear Regression**
- **Residual Analysis**

### 8. Bayesian Methods
- **Bayes' Theorem in Practice**
- **Bayesian Inference**
- **Prior Selection**
- **Credible Intervals**

### 9. Time Series Analysis Basics
- **Autocorrelation and Partial Autocorrelation**
- **Stationarity**: ADF Test, KPSS Test
- **Trends and Seasonality**

### 10. Effect Size and Power
- **Cohen's d**: Standardized effect size
- **Power Analysis**
- **Sample Size Determination**

## Key Formulas

### Mean and Variance
```
Mean (μ) = (Σx) / n
Variance (σ²) = Σ(x - μ)² / n
Standard Deviation (σ) = √Variance
Standard Error (SE) = σ / √n
```

### Normal Distribution
```
f(x) = (1 / (σ√(2π))) * e^(-(x-μ)² / (2σ²))
Z-score = (x - μ) / σ
```

### Bayes' Theorem
```
P(A|B) = P(B|A) * P(A) / P(B)
```

### Binomial Distribution
```
P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
Expected Value: E(X) = n*p
Variance: Var(X) = n*p*(1-p)
```

### Confidence Interval
```
CI = Sample Statistic ± (Critical Value * Standard Error)
95% CI: μ ± 1.96 * (σ / √n)
```

### Hypothesis Test (t-test)
```
t = (Sample Mean - Population Mean) / (Sample SE)
df = n - 1
```

## File Structure

```
03_Statistics_Probability/
├── README.md (this file)
├── notes/
│   ├── 01_descriptive_statistics.md
│   ├── 02_probability_fundamentals.md
│   ├── 03_distributions.md
│   ├── 04_hypothesis_testing.md
│   ├── 05_correlation_regression.md
│   ├── 06_bayesian_methods.md
│   └── 07_advanced_statistics.md
├── code_examples/
│   ├── 01_descriptive_statistics.py
│   ├── 02_probability_distributions.py
│   ├── 03_hypothesis_testing.py
│   ├── 04_correlation_regression.py
│   ├── 05_bayesian_methods.py
│   └── 06_visualization_statistics.py
├── exercises/
│   ├── README.md
│   ├── basic_exercises.md
│   └── advanced_exercises.md
└── projects/
    ├── 01_exploratory_data_analysis.md
    ├── 02_ab_testing_analysis.md
    └── 03_statistical_modeling_project.md
```

## Prerequisites
- Chapter 02: Python Basics (Data structures, control flow, functions)
- Python libraries: NumPy, Pandas, SciPy, Matplotlib, Seaborn
- Basic algebra and calculus understanding

## Learning Path

1. **Week 1-2**: Descriptive Statistics
   - Read: notes/01_descriptive_statistics.md
   - Code: code_examples/01_descriptive_statistics.py
   - Practice: exercises/basic_exercises.md

2. **Week 2-3**: Probability
   - Read: notes/02_probability_fundamentals.md
   - Read: notes/03_distributions.md
   - Code: code_examples/02_probability_distributions.py

3. **Week 3-4**: Hypothesis Testing
   - Read: notes/04_hypothesis_testing.md
   - Code: code_examples/03_hypothesis_testing.py
   - Project: projects/02_ab_testing_analysis.md

4. **Week 4-5**: Correlation & Regression
   - Read: notes/05_correlation_regression.md
   - Code: code_examples/04_correlation_regression.py
   - Project: projects/01_exploratory_data_analysis.md

5. **Week 5-6**: Advanced Topics
   - Read: notes/06_bayesian_methods.md
   - Read: notes/07_advanced_statistics.md
   - Code: code_examples/05_bayesian_methods.py

## Key Takeaways

✓ Statistics helps us understand data through summarization and inference
✓ Probability theory provides the mathematical foundation for uncertainty quantification
✓ Different distributions model different types of real-world phenomena
✓ Hypothesis testing allows us to make data-driven decisions
✓ Understanding statistical properties is essential for building robust ML models
✓ Effect size matters more than just statistical significance
✓ Bayesian methods offer alternative ways to think about probability and inference

## Common Mistakes to Avoid

❌ Confusing correlation with causation
❌ p-hacking and data dredging
❌ Ignoring assumptions of statistical tests
❌ Using the wrong test for your data type
❌ Ignoring effect size in favor of p-values
❌ Multiple comparisons without correction
❌ Extrapolating beyond the data range
❌ Assuming normality without checking

## Useful Resources

### Books
- "Statistical Rethinking" by Richard McElreath
- "The Book of Why" by Judea Pearl
- "Introduction to Statistical Learning" by James, Witten, Hastie, Tibshirani

### Online Resources
- Khan Academy: Statistics and Probability
- StatQuest with Josh Starmer (YouTube)
- Codecademy: Statistics Course
- Coursera: Statistics Specializations

### Python Libraries
- **SciPy**: Statistical functions and distributions
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Statsmodels**: Statistical models and tests
- **Scikit-learn**: Machine learning with statistical foundations

## Next Steps

After mastering this chapter:
- Move to Chapter 04: Data Preprocessing for data cleaning and preparation
- Explore Chapter 05: Linear Regression for predictive modeling
- Study Chapter 06: Logistic Regression for classification

## Questions to Test Your Understanding

1. What's the difference between population and sample statistics?
2. When should you use median vs. mean?
3. Explain conditional probability with an example
4. What does a p-value really mean?
5. How do you interpret a 95% confidence interval?
6. Why is Type I error more concerning than Type II in some contexts?
7. How does sample size affect hypothesis testing?
8. What are the assumptions of ANOVA?
9. Explain the difference between correlation and causation
10. When would you use Bayesian methods instead of frequentist?

---

**Last Updated**: December 2024
**Difficulty Level**: Intermediate
**Estimated Time**: 6-8 weeks
