#!/usr/bin/env python3
"""Comprehensive Statistics Examples

Covering descriptive statistics, probability distributions, and hypothesis testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ===== DESCRIPTIVE STATISTICS =====
def descriptive_statistics():
    """Demonstrate descriptive statistical measures."""
    print("\n=== Descriptive Statistics ===")
    
    # Sample data
    data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    
    # Central Tendency
    print(f"Data: {data}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Mode: {stats.mode(data, keepdims=True).mode[0]}")
    
    # Dispersion
    print(f"\nRange: {np.max(data) - np.min(data)}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Std Dev: {np.std(data):.2f}")
    print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    
    # Shape
    print(f"\nSkewness: {stats.skew(data):.2f}")
    print(f"Kurtosis: {stats.kurtosis(data):.2f}")
    
    # Describe function
    df = pd.DataFrame({'values': data})
    print(f"\nPandas Describe:\n{df.describe()}")

# ===== PROBABILITY DISTRIBUTIONS =====
def probability_distributions():
    """Demonstrate common probability distributions."""
    print("\n=== Probability Distributions ===")
    
    # Normal Distribution
    print("\n1. Normal Distribution")
    mean, std = 0, 1
    x = np.linspace(-4, 4, 100)
    y = stats.norm.pdf(x, mean, std)
    print(f"PDF at x=0: {stats.norm.pdf(0, mean, std):.4f}")
    print(f"CDF at x=0: {stats.norm.cdf(0, mean, std):.4f}")
    print(f"P(Z < 1.96): {stats.norm.cdf(1.96):.4f}")
    
    # Binomial Distribution
    print("\n2. Binomial Distribution")
    n, p = 10, 0.5
    binom_dist = stats.binom(n, p)
    print(f"P(X=5) when n=10, p=0.5: {binom_dist.pmf(5):.4f}")
    print(f"P(X<=5): {binom_dist.cdf(5):.4f}")
    print(f"Expected value: {binom_dist.mean():.2f}")
    print(f"Variance: {binom_dist.var():.2f}")
    
    # Poisson Distribution
    print("\n3. Poisson Distribution")
    lambda_param = 3
    poisson_dist = stats.poisson(lambda_param)
    print(f"P(X=3) when λ=3: {poisson_dist.pmf(3):.4f}")
    print(f"Mean: {poisson_dist.mean():.2f}")
    
    # Chi-square Distribution
    print("\n4. Chi-square Distribution")
    df = 2
    chi_dist = stats.chi2(df)
    print(f"PDF at x=1: {chi_dist.pdf(1):.4f}")
    print(f"P(X < 1): {chi_dist.cdf(1):.4f}")

# ===== CORRELATION & COVARIANCE =====
def correlation_analysis():
    """Demonstrate correlation and covariance analysis."""
    print("\n=== Correlation & Covariance ===")
    
    # Sample data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 6])
    
    # Covariance
    cov = np.cov(x, y)[0, 1]
    print(f"Covariance(X, Y): {cov:.2f}")
    
    # Pearson Correlation
    pearson_r, p_value = stats.pearsonr(x, y)
    print(f"Pearson r: {pearson_r:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Spearman Correlation
    spearman_r, p_value = stats.spearmanr(x, y)
    print(f"Spearman r: {spearman_r:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Correlation matrix
    data = np.array([x, y, x**2])
    df = pd.DataFrame(data.T, columns=['X', 'Y', 'X^2'])
    print(f"\nCorrelation Matrix:\n{df.corr()}")

# ===== HYPOTHESIS TESTING =====
def hypothesis_testing():
    """Demonstrate various hypothesis tests."""
    print("\n=== Hypothesis Testing ===")
    
    # One-sample t-test
    print("\n1. One-sample t-test")
    data = np.array([2.5, 3.2, 2.8, 3.5, 2.9])
    mu_0 = 3.0
    t_stat, p_value = stats.ttest_1samp(data, mu_0)
    print(f"H0: μ = {mu_0}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Reject H0: {p_value < 0.05}")
    
    # Two-sample t-test
    print("\n2. Two-sample t-test")
    sample1 = np.array([1, 2, 3, 4, 5])
    sample2 = np.array([2, 3, 4, 5, 6])
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Chi-square test
    print("\n3. Chi-square Goodness of Fit Test")
    observed = np.array([100, 150, 120, 130])
    expected = np.array([125, 125, 125, 125])
    chi2_stat, p_value = stats.chisquare(observed, expected)
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # ANOVA
    print("\n4. One-way ANOVA")
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])
    group3 = np.array([3, 4, 5, 6, 7])
    f_stat, p_value = stats.f_oneway(group1, group2, group3)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

# ===== CONFIDENCE INTERVALS =====
def confidence_intervals():
    """Demonstrate confidence interval calculation."""
    print("\n=== Confidence Intervals ===")
    
    data = np.array([10, 12, 11, 13, 12, 14, 11, 10])
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    
    # 95% Confidence Interval
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    print(f"Sample Mean: {mean:.2f}")
    print(f"Standard Error: {se:.2f}")
    print(f"95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
    
    # 99% Confidence Interval
    ci_99 = stats.t.interval(0.99, n-1, loc=mean, scale=se)
    print(f"99% CI: ({ci_99[0]:.2f}, {ci_99[1]:.2f})")

# ===== EFFECT SIZE =====
def effect_size():
    """Demonstrate effect size calculations."""
    print("\n=== Effect Size ===")
    
    # Cohen's d
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([3, 4, 5, 6, 7])
    
    mean_diff = np.mean(group2) - np.mean(group1)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    print(f"Group 1 Mean: {np.mean(group1):.2f}")
    print(f"Group 2 Mean: {np.mean(group2):.2f}")
    print(f"Cohen's d: {cohens_d:.2f}")
    print(f"Effect size: {'Small' if cohens_d < 0.5 else 'Medium' if cohens_d < 0.8 else 'Large'}")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("Comprehensive Statistics Examples")
    print("=" * 50)
    
    descriptive_statistics()
    probability_distributions()
    correlation_analysis()
    hypothesis_testing()
    confidence_intervals()
    effect_size()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
