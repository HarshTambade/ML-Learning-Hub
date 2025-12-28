# Probability Fundamentals

## Basic Concepts

### Probability Axioms
For any event A:
- P(A) ≥ 0 (non-negativity)
- P(S) = 1 (certainty, S = sample space)
- P(A ∪ B) = P(A) + P(B) if A and B are mutually exclusive

### Sample Space & Events
- **Sample Space (S)**: All possible outcomes
- **Event**: Subset of sample space
- **Complement**: A' = S - A, P(A') = 1 - P(A)

## Conditional Probability

### Definition
P(A|B) = P(A ∩ B) / P(B), where P(B) > 0

This means: "probability of A given that B has occurred"

### Chain Rule
P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)

### Example
Drawing cards without replacement:
- P(1st red) = 26/52
- P(2nd red | 1st red) = 25/51
- P(both red) = (26/52) × (25/51)

## Independence

### Definition
Events A and B are independent if:
P(A ∩ B) = P(A) × P(B)
OR equivalently: P(A|B) = P(A)

### Implications
- If A and B are independent, then:
  - P(A|B) = P(A)
  - P(B|A) = P(B)
  - Knowing B doesn't change probability of A

### Example
Flipping coins: P(H on flip 2) = 0.5 regardless of flip 1

## Bayes' Theorem

### Formula
P(A|B) = P(B|A) × P(A) / P(B)

Where:
- P(A|B) = Posterior probability
- P(B|A) = Likelihood
- P(A) = Prior probability
- P(B) = Evidence (total probability)

### Total Probability
If A₁, A₂, ..., Aₙ partition S:
P(B) = Σ P(B|Aᵢ) × P(Aᵢ)

### Full Form of Bayes' Theorem
P(Aₖ|B) = [P(B|Aₖ) × P(Aₖ)] / Σ[P(B|Aᵢ) × P(Aᵢ)]

### Medical Test Example
**Given:**
- Disease prevalence: P(D) = 0.01
- Test sensitivity: P(+|D) = 0.95
- Test specificity: P(-|¬D) = 0.99

**Find:** P(D|+) = ?

**Solution:**
- P(+|D) × P(D) = 0.95 × 0.01 = 0.0095
- P(+|¬D) × P(¬D) = 0.01 × 0.99 = 0.0099
- P(+) = 0.0095 + 0.0099 = 0.0194
- P(D|+) = 0.0095 / 0.0194 ≈ 0.49

## Counting Methods

### Multiplication Principle
If task 1 has n₁ ways and task 2 has n₂ ways:
Total ways = n₁ × n₂

### Permutations
Ordered arrangements of n items:
P(n,r) = n! / (n-r)!

Example: Arrange 3 people in line = P(3,3) = 6

### Combinations
Unordered selections of n items:
C(n,r) = n! / (r! × (n-r)!)

Example: Choose 2 from {A,B,C} = C(3,2) = 3

## Random Variables

### Definition
A function that assigns numbers to outcomes

### Types
- **Discrete**: Countable values (coin flips, dice)
- **Continuous**: Uncountable values (heights, times)

### Probability Distribution
- **PDF (Probability Density Function)**: For continuous variables
- **PMF (Probability Mass Function)**: For discrete variables
- **CDF (Cumulative Distribution Function)**: F(x) = P(X ≤ x)

## Expected Value & Variance

### Expected Value (Mean)
E(X) = Σ x × P(X = x) for discrete
E(X) = ∫ x × f(x) dx for continuous

### Variance
Var(X) = E[X²] - E[X]²
SD(X) = √Var(X)

### Properties
- E(aX + b) = aE(X) + b
- Var(aX + b) = a² × Var(X)
- E(X + Y) = E(X) + E(Y)
- Var(X + Y) = Var(X) + Var(Y) if X, Y independent

## Covariance & Correlation

### Covariance
Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]

Measures joint variability
- Positive: move together
- Negative: move opposite
- Zero: no linear relationship

### Correlation
r = Cov(X,Y) / (σₓ × σᵧ)

Standardized covariance:
- Range: -1 to +1
- Unitless
- Easier to interpret

## Common Probability Distributions

### Binomial
Number of successes in n independent trials:
- P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- E(X) = np
- Var(X) = np(1-p)

### Poisson
Number of events in fixed interval:
- P(X = k) = (e^-λ × λ^k) / k!
- E(X) = λ
- Var(X) = λ

### Normal
Most important distribution:
- Bell-shaped, symmetric
- Defined by μ and σ
- 68-95-99.7 rule

### Exponential
Time until next event:
- f(x) = λe^(-λx)
- E(X) = 1/λ
- Var(X) = 1/λ²

## Inequalities

### Markov's Inequality
For non-negative X:
P(X ≥ a) ≤ E(X) / a

### Chebyshev's Inequality
P(|X - μ| ≥ kσ) ≤ 1/k²

Example: At least 75% of data within 2σ of mean

## Law of Large Numbers

As sample size increases, sample mean converges to expected value:
lim(n→∞) (X₁ + X₂ + ... + Xₙ)/n = E(X)

## Central Limit Theorem

Sum/average of independent random variables approaches normal distribution:
Z = (X̄ - μ) / (σ/√n) → N(0,1)

True regardless of original distribution!

## Key Takeaways

✓ Probability quantifies uncertainty
✓ Conditional probability is crucial for inference
✓ Bayes' theorem updates beliefs with evidence
✓ Independence simplifies calculations
✓ Counting methods solve combinatorial problems
✓ Distributions model random phenomena
✓ Laws ensure sample statistics reflect population
