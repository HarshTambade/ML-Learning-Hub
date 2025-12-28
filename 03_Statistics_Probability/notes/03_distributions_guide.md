# Probability Distributions Guide

## Overview
Probability distributions model the behavior of random variables. Understanding distributions is essential for statistical inference and machine learning.

## Discrete Distributions

### Bernoulli Distribution
**Use**: Single trial with two outcomes (success/failure)

**Parameters**: p (success probability)

**PMF**: P(X=1) = p, P(X=0) = 1-p

**Properties**:
- E(X) = p
- Var(X) = p(1-p)

**Example**: Coin flip (p=0.5)

### Binomial Distribution
**Use**: Number of successes in n independent trials

**Parameters**: n (trials), p (success probability)

**PMF**: P(X=k) = C(n,k) × p^k × (1-p)^(n-k)

**Properties**:
- E(X) = np
- Var(X) = np(1-p)
- Normal approximation: X ~ N(np, np(1-p)) for large n

**Example**: Number of heads in 10 coin flips

### Poisson Distribution
**Use**: Number of events in fixed interval (time/space)

**Parameters**: λ (event rate)

**PMF**: P(X=k) = (e^-λ × λ^k) / k!

**Properties**:
- E(X) = λ
- Var(X) = λ
- Approximates Binomial when n large, p small, np moderate

**Example**: Emails received per hour (λ=5)

### Geometric Distribution
**Use**: Number of trials until first success

**Parameters**: p (success probability)

**PMF**: P(X=k) = (1-p)^(k-1) × p

**Properties**:
- E(X) = 1/p
- Var(X) = (1-p)/p²

**Example**: Dice rolls until first 6

### Hypergeometric Distribution
**Use**: Drawing without replacement from finite population

**Parameters**: N (population), K (success items), n (draws)

**PMF**: P(X=k) = [C(K,k) × C(N-K,n-k)] / C(N,n)

**Properties**:
- E(X) = n × K/N
- Var(X) = n × (K/N) × (1-K/N) × (N-n)/(N-1)

**Example**: Lottery drawings

## Continuous Distributions

### Uniform Distribution
**Use**: Equal probability over interval [a,b]

**Parameters**: a (lower), b (upper)

**PDF**: f(x) = 1/(b-a) for a ≤ x ≤ b

**Properties**:
- E(X) = (a+b)/2
- Var(X) = (b-a)²/12

**Example**: Random selection from [0,1]

### Normal (Gaussian) Distribution
**Use**: Most common distribution, natural phenomena

**Parameters**: μ (mean), σ (std dev)

**PDF**: f(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))

**Properties**:
- E(X) = μ
- Var(X) = σ²
- Symmetric about mean
- 68-95-99.7 rule

**Standard Normal Z ~ N(0,1)**:
- P(Z < 1.96) ≈ 0.975
- P(|Z| < 1.96) ≈ 0.95

**Example**: Heights, test scores, measurement errors

### Exponential Distribution
**Use**: Time between events (waiting time)

**Parameters**: λ (rate)

**PDF**: f(x) = λe^(-λx) for x ≥ 0

**Properties**:
- E(X) = 1/λ
- Var(X) = 1/λ²
- Memoryless property

**Example**: Customer service times

### Gamma Distribution
**Use**: Sum of exponential random variables

**Parameters**: α (shape), β (rate)

**PDF**: f(x) = (β^α / Γ(α)) × x^(α-1) × e^(-βx)

**Properties**:
- E(X) = α/β
- Var(X) = α/β²
- Exponential is special case (α=1)

**Example**: Total service time for multiple customers

### Beta Distribution
**Use**: Proportions, probabilities, bounded [0,1]

**Parameters**: α, β (shape parameters)

**PDF**: f(x) = (x^(α-1) × (1-x)^(β-1)) / B(α,β)

**Properties**:
- E(X) = α/(α+β)
- Var(X) = (αβ) / ((α+β)²(α+β+1))

**Example**: Bayesian priors for proportions

### Chi-Square Distribution
**Use**: Testing variance, goodness-of-fit

**Parameters**: k (degrees of freedom)

**PDF**: Related to Normal distribution

**Properties**:
- E(X) = k
- Var(X) = 2k
- Always positive

**Example**: Chi-square test statistics

### t-Distribution
**Use**: Inference with small samples, unknown variance

**Parameters**: df (degrees of freedom)

**PDF**: Similar to Normal but with heavier tails

**Properties**:
- E(X) = 0 (for df > 1)
- Var(X) = df/(df-2) (for df > 2)
- Approaches Normal as df → ∞

**Example**: t-tests, confidence intervals

### F-Distribution
**Use**: Comparing variances, ANOVA

**Parameters**: df1, df2 (numerator, denominator df)

**Properties**:
- Always positive
- Right-skewed

**Example**: ANOVA F-statistics

## Key Relationships

```
Bernoulli → Binomial (sum of Bernoullis)
Binomial → Poisson (n large, p small)
Binomial → Normal (n large)
Poisson → Normal (λ large)
Exponential → Gamma (sum of exponentials)
Multiple Normals → Chi-square
Normal / √(Chi-square/df) → t-distribution
(Chi-square/df1) / (Chi-square/df2) → F-distribution
```

## Choosing a Distribution

| Scenario | Distribution | Parameters |
|----------|---|---|
| Binary outcome | Bernoulli | p |
| Count successes | Binomial | n, p |
| Events over time | Poisson | λ |
| Time until event | Exponential | λ |
| Continuous, symmetric | Normal | μ, σ |
| Proportions/probabilities | Beta | α, β |
| Without replacement | Hypergeometric | N, K, n |
| Sum of exponentials | Gamma | α, β |

## Transformation & Relationships

### Location-Scale Transformations
If X ~ Dist(θ), then aX + b transforms the distribution

**Normal special case**:
- If X ~ N(μ, σ²), then Z = (X-μ)/σ ~ N(0,1)

### Log-Normal Distribution
If X ~ Lognormal, then ln(X) ~ Normal
- Useful for positive-skewed data

## Central Limit Theorem Connection

For any distribution with finite mean and variance:
- Sample mean approaches Normal distribution
- Holds for large sample sizes
- Justifies Normal approximations

## Key Takeaways

✓ Different distributions model different phenomena
✓ Discrete distributions count events
✓ Continuous distributions measure quantities
✓ Normal distribution is most important
✓ Relationships between distributions exist
✓ Choosing correct distribution is crucial
✓ Parameter estimation determines specific behavior
