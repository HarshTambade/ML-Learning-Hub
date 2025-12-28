# Chapter 03 Projects: Statistics & Probability

## Overview
These projects apply statistical concepts to real-world scenarios. Each project builds data analysis, visualization, and statistical inference skills.

## Project 1: Exploratory Data Analysis (EDA)

### Objective
Perform comprehensive EDA on a dataset, calculating descriptive statistics and creating visualizations.

### Requirements
1. **Data Loading**: Load dataset from CSV/Excel
2. **Descriptive Statistics**:
   - Calculate mean, median, std, min, max for all numeric variables
   - Identify distributions (normal, skewed, etc.)
   - Create correlation matrix
   - Detect and report outliers
3. **Visualizations**:
   - Histograms with KDE for each numeric variable
   - Boxplots for outlier detection
   - Heatmap of correlations
   - Scatter plots for interesting pairs
4. **Report**: Summary of key findings

### Dataset Suggestions
- Iris Dataset
- Boston Housing
- Kaggle: Titanic
- Your own dataset

### Deliverables
- Jupyter notebook with code and analysis
- PDF report with visualizations
- Statistical summary table

---

## Project 2: A/B Testing Analysis

### Objective
Design and analyze an A/B test to compare two conditions.

### Requirements
1. **Problem Definition**:
   - Clearly state hypothesis
   - Define control and treatment groups
   - Specify success metric
2. **Sample Size Calculation**:
   - Determine effect size of interest
   - Calculate sample size for power=0.80
   - Justify assumptions
3. **Data Collection**:
   - Simulate or collect real A/B test data
   - Ensure random assignment
   - Record baseline metrics
4. **Statistical Analysis**:
   - Check assumptions (normality, equal variance)
   - Perform appropriate test (t-test, chi-square, etc.)
   - Calculate confidence intervals
   - Report p-value and effect size (Cohen's d)
5. **Interpretation**:
   - Write clear conclusions
   - Discuss business implications
   - Recommend actions

### Real-World Examples
- Website button color (conversion rate)
- Email subject lines (open rate)
- Pricing strategy (revenue per user)
- App UI change (user engagement)

### Deliverables
- Hypothesis document
- Code for simulation/analysis
- Statistical test results
- Executive summary report

---

## Project 3: Hypothesis Testing on Real Data

### Objective
Perform multiple hypothesis tests on real data to answer research questions.

### Requirements
1. **Research Questions** (at least 3):
   - State null and alternative hypotheses
   - Specify significance level (α)
   - Choose appropriate test
2. **Assumption Checking**:
   - Verify test assumptions
   - Document violations
   - Use alternative tests if needed
3. **Statistical Testing**:
   - Perform parametric tests (t-test, ANOVA, correlation)
   - Perform non-parametric alternatives if needed
   - Report test statistics, p-values, degrees of freedom
4. **Multiple Comparisons**:
   - Apply Bonferroni correction if multiple tests
   - Use FDR control as alternative
5. **Effect Size**:
   - Report Cohen's d or R-squared
   - Interpret practical significance
6. **Visualization**:
   - Create plots showing distributions
   - Highlight significant differences
   - Show confidence intervals

### Example Research Questions
- Do male and female test scores differ significantly?
- Is there significant correlation between variables?
- Does treatment group differ from control across multiple measures?
- Are there significant differences among 3+ groups?

### Deliverables
- Research question document
- Assumption verification report
- Statistical analysis code
- Results table with statistics and interpretations
- Visualizations
- Comprehensive report

---

> **💡 Hypothesis Testing Reference**: For advanced examples of hypothesis testing techniques including ANOVA, Levene's test, Shapiro-Wilk normality test, and effect size calculations, see [`code_examples/02_hypothesis_testing_advanced.py`](../../code_examples/02_hypothesis_testing_advanced.py)

## Project 4: Regression Analysis

### Objective
Build and interpret regression models to predict or understand relationships.

### Requirements
1. **Exploratory Analysis**:
   - Correlation analysis
   - Scatter plots with trend lines
   - Identify potential predictors
2. **Model Building**:
   - Simple linear regression (one predictor)
   - Multiple linear regression (multiple predictors)
   - Model comparison and selection
3. **Model Validation**:
   - Check assumptions:
     - Linearity
     - Normality of residuals
     - Homogeneity of variance
     - Independence
   - Identify outliers and influential points
   - Calculate R-squared and adjusted R-squared
4. **Interpretation**:
   - Explain coefficients
   - Make predictions
   - Discuss limitations
5. **Visualization**:
   - Regression plots
   - Residual plots
   - Q-Q plots for normality
   - Scale-location plots

### Deliverables
- EDA report
- Regression analysis code
- Model comparison table
- Assumption checking plots
- Prediction examples
- Final report

---

## Project 5: Bayesian Analysis

### Objective
Apply Bayesian methods to estimate parameters and make inferences.

### Requirements
1. **Prior Specification**:
   - Choose prior distributions
   - Justify choices
   - Conduct sensitivity analysis
2. **Data and Likelihood**:
   - Specify likelihood function
   - Collect or simulate data
3. **Posterior Inference**:
   - Calculate posterior distribution
   - Use conjugate priors if possible
   - Or use MCMC methods for complex models
4. **Credible Intervals**:
   - Calculate 95% credible intervals
   - Compare with frequentist CI
5. **Visualization**:
   - Plot prior, likelihood, posterior
   - Posterior predictive distribution
   - Trace plots for MCMC convergence

### Topics
- Estimating success probability
- Comparing two proportions
- Hierarchical modeling

### Deliverables
- Problem definition
- Prior justification
- Analysis code
- Posterior inference results
- Comparison with frequentist approach
- Report

---

## Project 6: Power Analysis and Sample Size

### Objective
Design studies with adequate statistical power.

### Requirements
1. **Research Design**:
   - Define primary outcome
   - Specify expected effect size
   - Determine acceptable error rates
2. **Power Calculations**:
   - Calculate sample size for power=0.80, 0.90
   - Explore effect of varying parameters
   - Create visualization of power curves
3. **Cost-Benefit Analysis**:
   - Trade-off between power and sample size
   - Consider practical constraints
   - Make recommendation
4. **Sensitivity Analysis**:
   - How robust to violations of assumptions?
   - Impact of smaller than expected effect

### Deliverables
- Study design document
- Power calculation code
- Sample size table
- Power curves plots
- Recommendation and justification

---

## General Project Guidelines

### Code Quality
- Use version control (Git)
- Write clear, commented code
- Use functions for reusable code
- Include docstrings
- Follow PEP 8 style guide

### Report Structure
1. **Introduction**: Context and research questions
2. **Methods**: Data, variables, statistical techniques
3. **Results**: Findings with appropriate statistics
4. **Discussion**: Interpretation and implications
5. **Conclusion**: Summary and recommendations
6. **References**: Data sources, literature

### Reproducibility
- Include all data (or instructions to get it)
- Provide complete code
- Set random seeds for reproducibility
- Document software versions

### Visualization Best Practices
- Clear titles and axis labels
- Use appropriate chart types
- Include error bars/confidence intervals
- Use color-blind friendly palettes
- Provide figure captions

---

## Resources

### Data Sources
- Kaggle: https://www.kaggle.com/
- UCI ML Repository: https://archive.ics.uci.edu/ml/
- FiveThirtyEight: https://data.fivethirtyeight.com/
- Government Data: https://www.data.gov/

### Statistical Software
- Python: SciPy, Statsmodels, Scikit-learn
- R: Base R, tidyverse, ggplot2
- Online calculators for power analysis

### Learning Resources
- Khan Academy: Statistics
- StatQuest with Josh Starmer: YouTube
- Coursera: Statistical courses
- Textbooks: ISL, STAT110

---

## Evaluation Criteria

Each project will be evaluated on:
- **Analysis Quality** (40%): Appropriate methods, correct calculations
- **Interpretation** (30%): Clear explanation of results and implications
- **Code Quality** (15%): Clean, documented, reproducible
- **Presentation** (15%): Report clarity and visualizations

---

## Project Timeline

**Typical project duration: 1-2 weeks**

- Days 1-2: Plan and explore data
- Days 3-5: Conduct analysis
- Days 6-7: Visualize and interpret
- Days 8-10: Write report and finalize

---

## Questions?

Refer to exercises, notes, and code examples in this chapter.
Consult your instructor or peers for clarification.
