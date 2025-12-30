# Mathematical Foundations of Linear Regression

## Overview
Linear regression is a fundamental machine learning algorithm based on solid mathematical principles. This note covers the mathematical foundations necessary to understand how linear regression works.

## 1. The Linear Regression Model

### Simple Linear Regression
The simplest form of linear regression models the relationship between two variables:

```
y = β₀ + β₁x + ε
```

Where:
- **y**: Dependent variable (output)
- **x**: Independent variable (input)
- **β₀**: Intercept (y-value when x=0)
- **β₁**: Slope (rate of change)
- **ε**: Error term (residual)

### Multiple Linear Regression
Extends to multiple features:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

In matrix form:
```
Y = Xβ + ε
```

## 2. Cost Function (Loss Function)

### Mean Squared Error (MSE)
The objective is to minimize the sum of squared errors:

```
J(β) = (1/2m) * Σ(ŷᵢ - yᵢ)²
```

Where:
- **m**: Number of training examples
- **ŷᵢ**: Predicted value
- **yᵢ**: Actual value

### Why Square the Errors?
1. Penalizes large errors more heavily
2. Differentiable everywhere
3. Convex function (single minimum)
4. More mathematically tractable

## 3. Solution Methods

### Normal Equation (Closed-form Solution)
Direct calculation of optimal parameters:

```
β = (XᵀX)⁻¹Xᵀy
```

**Advantages:**
- Exact solution (no iterations)
- Simple implementation
- O(n³) complexity

**Disadvantages:**
- Slow for large datasets
- Matrix inversion required
- Numerically unstable if XᵀX is ill-conditioned

### Gradient Descent
Iterative optimization algorithm:

```
β := β - α * ∂J/∂β
```

Where α is the learning rate.

**Advantages:**
- Works with large datasets
- No matrix inversion needed
- Parallelizable

**Disadvantages:**
- Requires learning rate tuning
- Convergence not guaranteed
- May converge to local minima (though rare for linear regression)

## 4. Key Assumptions

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

## 5. Model Evaluation Metrics

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```

Where:
- **SS_res**: Sum of squared residuals
- **SS_tot**: Total sum of squares

Interpretation:
- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- R² < 0: Model worse than mean

### Mean Squared Error (MSE)
```
MSE = (1/m) * Σ(ŷᵢ - yᵢ)²
```

### Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```

More interpretable (same units as y).

## 6. Regularization

### Ridge Regression (L2 Regularization)
```
J(β) = MSE + λ * Σβⱼ²
```

Shrinks coefficients, reduces model complexity.

### Lasso Regression (L1 Regularization)
```
J(β) = MSE + λ * Σ|βⱼ|
```

Performs feature selection (some β become exactly 0).

### Elastic Net (L1 + L2)
```
J(β) = MSE + λ₁ * Σ|βⱼ| + λ₂ * Σβⱼ²
```

Combines benefits of Ridge and Lasso.

## 7. Mathematical Properties

### Convexity
The cost function is convex, guaranteeing a global minimum.

### Gradient
```
∂J/∂β = (1/m) * Xᵀ(Xβ - y)
```

### Hessian (Second derivative)
```
∂²J/∂β² = (1/m) * XᵀX
```

Always positive semi-definite (confirms convexity).

## 8. Bias-Variance Tradeoff

### Bias
- Error from overly simplistic model
- Underfitting: high bias

### Variance
- Error from model sensitivity to training data
- Overfitting: high variance

### Total Error
```
Total Error = Bias² + Variance + Irreducible Error
```

## 9. Generalization and Overfitting

### Training Error
Error on training data: may be artificially low

### Validation/Test Error
Error on unseen data: true measure of generalization

### Regularization Effect
- Increases training error slightly
- Decreases test error significantly
- Improves generalization

## Summary

Linear regression relies on:
- Simple but powerful linear model
- Well-defined optimization problem
- Multiple solution approaches
- Clear evaluation metrics
- Theoretical guarantees of optimality

Understanding these mathematical foundations is crucial for applying linear regression effectively and understanding its limitations.
