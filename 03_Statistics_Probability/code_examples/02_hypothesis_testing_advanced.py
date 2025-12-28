#!/usr/bin/env python3
"""Advanced Hypothesis Testing and Statistical Inference Examples

Covering ANOVA, regression, power analysis, and advanced testing techniques.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

# ===== ONE-WAY ANOVA =====
def anova_example():
    """Demonstrate one-way ANOVA for comparing multiple groups."""
    print("\n=== One-Way ANOVA ===")
    
    # Sample data: Test scores for 3 teaching methods
    method_a = np.array([72, 75, 78, 80, 76, 74, 79])
    method_b = np.array([68, 70, 72, 69, 71, 67, 70])
    method_c = np.array([85, 88, 82, 86, 89, 84, 87])
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(method_a, method_b, method_c)
    
    print(f"Group A mean: {method_a.mean():.2f}")
    print(f"Group B mean: {method_b.mean():.2f}")
    print(f"Group C mean: {method_c.mean():.2f}")
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Reject H0 (significant difference): {p_value < 0.05}")

# ===== LEVENE'S TEST FOR EQUALITY OF VARIANCES =====
def levene_test():
    """Test assumption of equal variances."""
    print("\n=== Levene's Test (Equality of Variances) ===")
    
    group1 = np.array([10, 12, 11, 13, 12])
    group2 = np.array([20, 25, 18, 22, 20])
    
    stat, p_value = stats.levene(group1, group2)
    print(f"Levene statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Equal variances (H0 true): {p_value > 0.05}")

# ===== NORMALITY TESTS =====
def normality_tests():
    """Test for normality assumption."""
    print("\n=== Normality Tests ===")
    
    # Normal data
    normal_data = np.random.normal(100, 15, 100)
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(normal_data)
    print(f"\nShapiro-Wilk test (normal data):")
    print(f"Statistic: {shapiro_stat:.4f}, P-value: {shapiro_p:.4f}")
    print(f"Is normal: {shapiro_p > 0.05}")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(normal_data, 'norm', args=(100, 15))
    print(f"\nKolmogorov-Smirnov test:")
    print(f"Statistic: {ks_stat:.4f}, P-value: {ks_p:.4f}")
    
    # Anderson-Darling test
    anderson_result = stats.anderson(normal_data)
    print(f"\nAnderson-Darling test:")
    print(f"Statistic: {anderson_result.statistic:.4f}")
    print(f"Critical values: {anderson_result.critical_values}")

# ===== NON-PARAMETRIC TESTS =====
def nonparametric_tests():
    """Demonstrate non-parametric alternatives."""
    print("\n=== Non-Parametric Tests ===")
    
    # Mann-Whitney U test (non-parametric alternative to t-test)
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 4, 6, 8, 10])
    
    stat, p_value = mannwhitneyu(group1, group2)
    print(f"\nMann-Whitney U test:")
    print(f"U-statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Wilcoxon signed-rank test (paired data)
    differences = np.array([1.2, -0.5, 0.8, 1.5, -0.3, 0.9])
    w_stat, w_p = stats.wilcoxon(differences)
    print(f"\nWilcoxon signed-rank test:")
    print(f"W-statistic: {w_stat:.4f}")
    print(f"P-value: {w_p:.4f}")
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    a = np.array([1, 2, 3, 4])
    b = np.array([4, 5, 6, 7])
    c = np.array([7, 8, 9, 10])
    h_stat, h_p = stats.kruskal(a, b, c)
    print(f"\nKruskal-Wallis test:")
    print(f"H-statistic: {h_stat:.4f}")
    print(f"P-value: {h_p:.4f}")

# ===== POST-HOC TESTS =====
def post_hoc_analysis():
    """Demonstrate post-hoc analysis for multiple comparisons."""
    print("\n=== Post-Hoc Analysis (Multiple Comparisons) ===")
    
    # After ANOVA, perform pairwise t-tests with Bonferroni correction
    groups = {
        'A': [72, 75, 78, 80],
        'B': [68, 70, 72, 69],
        'C': [85, 88, 82, 86]
    }
    
    # Bonferroni correction
    num_comparisons = 3  # A vs B, A vs C, B vs C
    bonferroni_alpha = 0.05 / num_comparisons
    
    print(f"Original alpha: 0.05")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    print(f"Number of comparisons: {num_comparisons}")
    
    # Perform pairwise comparisons
    t_ab, p_ab = ttest_ind(groups['A'], groups['B'])
    t_ac, p_ac = ttest_ind(groups['A'], groups['C'])
    t_bc, p_bc = ttest_ind(groups['B'], groups['C'])
    
    print(f"\nA vs B: t={t_ab:.4f}, p={p_ab:.4f}, significant={p_ab < bonferroni_alpha}")
    print(f"A vs C: t={t_ac:.4f}, p={p_ac:.4f}, significant={p_ac < bonferroni_alpha}")
    print(f"B vs C: t={t_bc:.4f}, p={p_bc:.4f}, significant={p_bc < bonferroni_alpha}")

# ===== POWER ANALYSIS =====
def power_analysis():
    """Demonstrate power analysis for study design."""
    print("\n=== Power Analysis ===")
    
    from scipy.stats import nct
    
    # Example: Calculating power for t-test
    # Given: effect size (Cohen's d), alpha, sample size
    effect_size = 0.5  # Medium effect
    alpha = 0.05
    n = 64  # sample size per group
    
    # Non-centrality parameter
    delta = effect_size * np.sqrt(n/2)
    
    # Critical t-value for two-tailed test
    df = 2 * n - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Power = 1 - beta = P(T > t_crit | H1 true)
    power = 1 - stats.nct.cdf(t_crit, df, delta)
    
    print(f"Effect size (Cohen's d): {effect_size}")
    print(f"Sample size per group: {n}")
    print(f"Alpha (Type I error): {alpha}")
    print(f"Power (1 - beta): {power:.4f}")
    print(f"Power percentage: {power*100:.2f}%")

# ===== CONFIDENCE INTERVALS =====
def confidence_intervals():
    """Calculate and interpret confidence intervals."""
    print("\n=== Confidence Intervals ===")
    
    data = np.array([10, 12, 11, 13, 12, 14, 11, 10, 13, 12])
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)  # Standard error
    
    # 95% CI using t-distribution
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    
    # 99% CI using t-distribution
    ci_99 = stats.t.interval(0.99, n-1, loc=mean, scale=se)
    
    print(f"Sample mean: {mean:.2f}")
    print(f"Standard error: {se:.4f}")
    print(f"95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
    print(f"99% CI: ({ci_99[0]:.2f}, {ci_99[1]:.2f})")

# ===== EFFECT SIZE CALCULATIONS =====
def effect_sizes():
    """Calculate different effect size measures."""
    print("\n=== Effect Size Measures ===")
    
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([3, 4, 5, 6, 7])
    
    # Cohen's d
    mean_diff = group2.mean() - group1.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    # Pearson's r
    r, r_p = stats.pearsonr(group1, group2)
    
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}")
    print(f"\nPearson's r: {r:.4f}")
    print(f"R-squared (variance explained): {r**2:.4f}")

# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("Advanced Hypothesis Testing and Statistical Inference")
    print("=" * 60)
    
    anova_example()
    levene_test()
    normality_tests()
    nonparametric_tests()
    post_hoc_analysis()
    power_analysis()
    confidence_intervals()
    effect_sizes()
    
    print("\n" + "=" * 60)
    print("All hypothesis testing examples completed!")
