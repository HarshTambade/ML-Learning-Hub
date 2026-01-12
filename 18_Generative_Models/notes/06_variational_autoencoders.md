# Variational Autoencoders (VAEs)

## Overview

Variational Autoencoders (VAEs) are a powerful class of generative models that combine deep learning with Bayesian inference. Unlike standard autoencoders, VAEs provide a principled probabilistic framework for generating new data samples.

### Key Characteristics

- **Latent Variable Model**: Learn a latent code to represent data
- **Probabilistic Framework**: Model probability distributions over data
- **Encoder-Decoder Architecture**: Two neural networks working together
- **ELBO Optimization**: Maximize Evidence Lower Bound
- **Reparameterization Trick**: Enable gradient-based training

## Mathematical Foundation

### The Generative Model

VAEs model the data as generated from a latent variable distribution:

```
P(X) = ∫ P(X|z)P(z) dz
```

Where:
- **z**: Latent variables (typically N(0, I))
- **P(X|z)**: Decoder - likelihood of data given latent code
- **P(z)**: Prior - distribution of latent variables

### The Evidence Lower Bound (ELBO)

The key to training VAEs is the ELBO, which provides a tractable lower bound on the log-likelihood:

```
log P(X) ≥ ELBO = E_q(z|X)[log P(X|z)] - KL(q(z|X)||P(z))
```

**ELBO Components:**

1. **Reconstruction Term**: E_q(z|X)[log P(X|z)]
   - Measures how well the decoder reconstructs X from z
   - Also called "reconstruction loss" or "likelihood term"
   - Encourages learned features to capture important data characteristics

2. **KL Regularization Term**: KL(q(z|X)||P(z))
   - Measures how much the learned posterior q(z|X) diverges from prior P(z)
   - Encourages latent space to follow prior distribution
   - Prevents posterior collapse
   - Acts as regularization for smooth latent space

### Kullback-Leibler Divergence

The KL divergence measures distance between two probability distributions:

```
KL(Q||P) = ∑_z Q(z) log(Q(z)/P(z))
         = E_Q[log Q(z) - log P(z)]
```

**Properties:**
- KL(Q||P) ≥ 0, with equality iff Q = P
- NOT symmetric: KL(Q||P) ≠ KL(P||Q)
- Asymptotically equivalent to chi-squared statistic

### Gaussian VAE Formulation

For Gaussian encoder/decoder:

```
q(z|x) = N(μ_encoder(x), σ_encoder(x)²)
P(x|z) = N(μ_decoder(z), σ_decoder(z)²)
P(z) = N(0, I)

KL(q(z|x)||N(0,I)) = 1/2 ∑_j (μ_j² + σ_j² - log σ_j² - 1)
```

## Architecture Components

### Encoder Network

The encoder maps input data x to latent distribution parameters:

```
q(z|x) = N(μ_encoder(x), σ_encoder²(x))
```

**Architecture Example:**
```
Input (784) → Dense (400) → ReLU → Dense (400) → ReLU 
  → Dense (20) [μ] and Dense (20) [log σ²]
```

**Output:**
- Mean vector μ: Center of latent distribution
- Log-variance vector log σ²: Spread of distribution

### Reparameterization Trick

To sample z while maintaining differentiability:

```
z = μ + σ ⊙ ε, where ε ~ N(0, I)
```

Where ⊙ denotes element-wise multiplication.

**Why This Works:**
- Allows gradient flow through stochastic sampling
- Backpropagation through "reparameterization" instead of sampling operation
- Enables end-to-end training with SGD

### Decoder Network

The decoder maps latent code z back to data space:

```
P(x|z) = N(μ_decoder(z), σ_decoder²(z))
```

**Architecture Example:**
```
Input (20) → Dense (400) → ReLU → Dense (400) → ReLU 
  → Dense (784) [μ] and Dense (784) [log σ²]
```

## Training Process

### Loss Function

```
L_VAE = -E_q(z|x)[log P(x|z)] + KL(q(z|x)||P(z))
      = Reconstruction_Loss + KL_Divergence
```

### Training Algorithm

1. **Forward Pass:**
   - Encode x → get μ, log σ²
   - Sample ε ~ N(0, I)
   - Reparameterize: z = μ + σ ⊙ ε
   - Decode z → get reconstruction x'

2. **Loss Computation:**
   - Reconstruction loss: ||x - x'||²
   - KL divergence: 1/2 ∑ (μ² + σ² - log σ² - 1)

3. **Backward Pass:**
   - Gradient through decoder
   - Gradient through reparameterization
   - Gradient through encoder
   - Update parameters

## Key Hyperparameters & Techniques

### β-VAE (Disentangled Representations)

Modify the loss to emphasize KL term:

```
L_β = -E_q(z|x)[log P(x|z)] + β·KL(q(z|x)||P(z))
```

- β > 1: Emphasize KL regularization (more disentangled, blurrier reconstructions)
- β < 1: Emphasize reconstruction (less constrained, entangled features)
- β = 1: Standard VAE

### KL Annealing

Gradually increase KL weight during training:

```
β(t) = β_final · sigmoid((t - t_threshold) / rate)
```

**Motivation:**
- Prevents "posterior collapse" early in training
- Allows encoder to learn useful features first
- Gradually enforce distributional constraints

### Free Bits

Keep KL divergence above a minimum:

```
L = Reconstruction + max(κ, KL_divergence)
```

Where κ is a threshold (typically 0.25-1.0 nats).

## Important Concepts & Phenomena

### Posterior Collapse

**Problem**: KL divergence → 0, posterior ignores data (q(z|x) ≈ P(z))

**Consequences:**
- Encoder becomes unused
- Decoder generates from prior alone
- Model becomes standard autoencoder

**Solutions:**
- KL annealing
- Free bits strategy
- β-VAE with β > 1
- Increased latent dimension
- Architectural modifications

### Blurry Reconstructions

**Why**: VAE trades off reconstruction quality for clean latent space

**Mathematical reason**: Gaussian decoder assumes pixel-level uncertainty

**Solutions:**
- VQ-VAE: Discrete latent codes
- Hierarchical VAE: Multi-scale latents
- Alternative decoders: Bernoulli for binary data

## Applications

### 1. Data Generation

**Sample from Prior:**
```
z ~ N(0, I)
x = decoder(z)
```

Generates diverse new samples from learned distribution.

### 2. Latent Space Interpolation

```
z_interpolated = (1-t)·z1 + t·z2, t ∈ [0,1]
x_interpolated = decoder(z_interpolated)
```

Smooth transitions between data points.

### 3. Dimensionality Reduction

```
z_encoded = encoder(x)
```

Compresses high-dimensional data to latent codes.

### 4. Semi-Supervised Learning

Use unlabeled data to learn representations, then fine-tune classifier.

### 5. Anomaly Detection

```
reconstruction_error = ||x - decoder(encoder(x))||
```

Out-of-distribution samples have high reconstruction error.

## Variants of VAEs

### Conditional VAE (CVAE)

Generate data conditioned on class/attribute:

```
q(z|x, y) and P(x|z, y)
```

Useful for class-specific generation.

### Hierarchical VAE

Multiple levels of latent variables:

```
z1 ~ P(z1)
z2 ~ P(z2|z1)
x ~ P(x|z1,z2)
```

Captures multi-scale structure.

### Vector Quantized VAE (VQ-VAE)

Discrete latent codes instead of continuous:

```
e(x) → nearest codebook vector → z
```

Improves reconstruction quality.

### Adversarial Autoencoders

Combine VAE with adversarial training for better latent distribution match.

## Comparison with Related Models

### VAE vs Standard Autoencoder

| Aspect | VAE | Standard AE |
|--------|-----|-------------|
| Probabilistic | Yes | No |
| Latent Distribution | Constrained | Unstructured |
| Generation Quality | Moderate | N/A |
| Disentanglement | Often good | Not enforced |
| Training Stability | Stable | More stable |

### VAE vs GAN

| Aspect | VAE | GAN |
|--------|-----|-----|
| Training | Stable, ELBO | Unstable, minimax |
| Mode Coverage | Good | Can collapse |
| Sample Quality | Good but blurry | Excellent but diverse |
| Likelihood | Tractable | Not tractable |
| Encoder | Yes | No |
| Interpretability | High | Low |

## Implementation Tips

### Network Design
- Use BatchNorm for stability
- Start with small latent dimension
- Gradually increase during training
- Use ReLU activations

### Hyperparameter Tuning
- Learning rate: 0.001-0.01
- Batch size: 32-128
- Latent dim: 10-100 for images
- β: Start at 1, adjust based on collapse

### Training Monitoring
- Track reconstruction loss (should decrease)
- Track KL divergence (should decrease)
- Monitor ELBO (should increase)
- Visualize latent space (t-SNE)
- Check reconstruction samples

## Mathematical Details: Evidence Lower Bound Derivation

**Starting from Jensen's Inequality:**
```
log P(x) = log ∫ P(x|z)P(z) dz
         = log ∫ P(x|z)P(z)/q(z|x) · q(z|x) dz
         = log E_q[(P(x|z)P(z)/q(z|x))]
         ≥ E_q[log(P(x|z)P(z)/q(z|x))]  [Jensen's ineq]
         = E_q[log P(x|z)] + E_q[log P(z)] - E_q[log q(z|x)]
         = E_q[log P(x|z)] - KL(q(z|x)||P(z))
```

The inequality becomes equality when q(z|x) = P(z|x).

## Practical Considerations

### Computational Complexity
- Forward pass: O(n·d·h) where d=latent dim, h=hidden units
- Backward pass: Same complexity
- Memory: Linear in batch size and architecture

### Scalability
- Works well for images up to 256×256
- For larger images: Use hierarchical VAE or progressive training
- For variable-length sequences: Use RNN encoder/decoder

## Conclusion

Variational Autoencoders provide an elegant framework combining:
- Probabilistic modeling
- Deep neural networks
- Bayesian inference

This makes them invaluable for:
- Unsupervised learning
- Semi-supervised learning
- Data generation
- Representation learning

Their main limitation (blurry reconstructions) can be addressed with careful architecture design and variant modifications.
