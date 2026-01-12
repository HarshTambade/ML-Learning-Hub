# Generative Models Fundamentals

## Overview
Generative models are a class of machine learning models that learn the underlying probability distribution of data and can generate new data samples from this distribution.

## Key Concepts

### 1. Probability Distribution Learning
- Generative models aim to model P(X) - the probability distribution of data
- Learn the underlying structure and patterns in the data
- Can generate novel samples that follow the learned distribution

### 2. Discriminative vs Generative Models

**Discriminative Models:**
- Learn P(Y|X) - probability of output given input
- Focused on decision boundaries
- Examples: CNNs, Linear Classifiers
- Better for classification tasks

**Generative Models:**
- Learn P(X) or P(X,Y) - joint or marginal distribution
- Understand data structure
- Can generate new data
- Examples: GANs, VAEs, Autoencoders

### 3. Types of Generative Models

#### a) Autoregressive Models
- Generate data one element at a time
- Each element depends on previous elements
- Example: PixelCNN, WaveNet
- Formula: P(X) = ∏ P(x_i | x_1:i-1)

#### b) Latent Variable Models
- Use hidden/latent variables z to generate data
- P(X) = ∫ P(X|z)P(z) dz
- Examples: VAEs, GANs, Autoencoders

#### c) Energy-Based Models
- Model data using energy functions
- P(X) = exp(-E(X)) / Z
- Z is normalization constant

#### d) Flow-Based Models
- Use invertible transformations
- Exact likelihood computation
- Examples: Glow, RealNVP

### 4. Applications of Generative Models

1. **Image Generation**
   - Create photorealistic images
   - Image inpainting and super-resolution
   - Face synthesis and enhancement

2. **Data Augmentation**
   - Generate synthetic training data
   - Address class imbalance
   - Improve model generalization

3. **Anomaly Detection**
   - Identify samples far from learned distribution
   - Reconstruction error as anomaly score

4. **Representation Learning**
   - Learn meaningful latent representations
   - Dimensionality reduction
   - Feature extraction

5. **Text Generation**
   - Language models
   - Machine translation
   - Text summarization

6. **Drug Discovery**
   - Generate novel molecular structures
   - Protein generation

### 5. Evaluation Metrics

#### Likelihood-Based Metrics
- **Log-Likelihood**: Measures how well model explains data
- **Perplexity**: Exponential of negative log-likelihood

#### Sample Quality Metrics
- **Inception Score (IS)**: Quality and diversity of generated samples
- **Fréchet Inception Distance (FID)**: Distance between real and generated distributions
- **Kernel Inception Distance (KID)**: Kernel version of FID

#### Inception Score Formula:
IS(G) = exp(E_x[KL(p(y|x) || p(y))])

Where:
- p(y|x) = class probability for generated image x
- p(y) = marginal class distribution

#### FID Formula:
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))

Where:
- μ = mean of inception features
- Σ = covariance of inception features
- r = real, g = generated

### 6. Loss Functions

#### Reconstruction Loss
- MSE Loss: L = ||X - X_recon||²
- BCE Loss: L = -Σ[y log(ŷ) + (1-y)log(1-ŷ)]

#### KL Divergence
- Measures divergence between two distributions
- KL(P||Q) = Σ P(x)log(P(x)/Q(x))
- Used in VAEs and flow models

#### Wasserstein Distance
- Earth Mover's Distance
- W(P,Q) = inf_γ E_{(x,y)~γ}[||x-y||]
- More stable than JS divergence

### 7. Training Challenges

1. **Mode Collapse**
   - Generator produces limited diversity
   - Common in GANs
   - Solution: Spectral normalization, Progressive Growing

2. **Vanishing Gradients**
   - Gradients become too small
   - Difficult for training
   - Solution: Wasserstein distance, Gradient penalty

3. **Unstable Training**
   - Loss oscillates
   - Poor convergence
   - Solution: Better architectures, Careful hyperparameters

4. **Computational Cost**
   - Generative models are resource-intensive
   - Solution: Efficient architectures, Distributed training

### 8. Sampling Methods

#### Ancestral Sampling
1. Sample z ~ P(z)
2. Generate X = G(z)
3. Simple but requires full forward pass

#### Importance Sampling
- Weight samples by importance ratio
- Better for approximating expectations

#### Markov Chain Monte Carlo (MCMC)
- Metropolis-Hastings algorithm
- Gibbs sampling
- Slow but asymptotically unbiased

### 9. Mathematical Foundations

#### Bayes' Rule
P(Z|X) = P(X|Z)P(Z) / P(X)

#### Variational Inference
Used when posterior is intractable:
Q(Z|X) approximates P(Z|X)
ELBO: L ≥ log P(X)
L = E_Q[log P(X|Z)] - KL(Q||P)

#### Jensen's Inequality
For concave function f:
f(E[X]) ≥ E[f(X)]
Used to derive ELBO

## Summary

Generative models are powerful tools for:
- Understanding data distributions
- Generating new data samples
- Feature learning
- Data augmentation
- Multiple applications across domains

Key trade-offs:
- Complexity vs Training Stability
- Sample Quality vs Diversity
- Computational Cost vs Model Capacity
- Likelihood vs Sample Quality
