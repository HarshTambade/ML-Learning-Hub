# GANs: Generative Adversarial Networks - Detailed Guide

## Overview
GANs are a framework for training generative models through adversarial process between two networks: a generator and a discriminator.

## Architecture

### 1. Generator Network (G)
- Takes random noise z as input
- Generates fake data samples: x_fake = G(z)
- Goal: Fool the discriminator
- Architecture: Usually transposed convolutions (for images)

### 2. Discriminator Network (D)
- Takes real or fake data as input
- Outputs probability that input is real: D(x) ∈ [0,1]
- Goal: Distinguish real from fake
- Architecture: Usually standard convolutions + classifier

## Training Process

### Minimax Game
- Generator and discriminator play a two-player zero-sum game
- Objective function:
  V(D,G) = E_x~pdata[log D(x)] + E_z~pz[log(1-D(G(z)))]

### Training Procedure
1. **Train Discriminator:**
   - Maximize: log D(x) + log(1-D(G(z)))
   - Uses real and fake batches
   - Updates discriminator parameters

2. **Train Generator:**
   - Maximize: log D(G(z))
   - Or minimize: log(1-D(G(z)))
   - Updates generator parameters only

### Alternating Optimization
```
for each training iteration:
    - Sample batch of real data and noise
    - Update discriminator
    - Sample new batch of noise  
    - Update generator
    - Repeat
```

## Loss Functions

### Standard GAN Loss
- Discriminator: L_D = -E_x[log D(x)] - E_z[log(1-D(G(z)))]
- Generator: L_G = -E_z[log D(G(z))]
- Problem: Vanishing gradients early in training

### Wasserstein GAN (WGAN) Loss
- Uses Wasserstein distance instead of JS divergence
- Discriminator (Critic): max E_x[D(x)] - E_z[D(G(z))]
- Generator: min E_z[D(G(z))]
- Addresses vanishing gradient problem
- Requires Lipschitz constraint (weight clipping)

### Least Squares GAN (LSGAN)
- Discriminator: 0.5 * E_x[(D(x)-1)²] + 0.5 * E_z[D(G(z))²]
- Generator: 0.5 * E_z[(D(G(z))-1)²]
- More stable training
- Better gradient flow

## Key Challenges

### 1. Mode Collapse
- Generator learns to produce limited variety
- Produces same/similar samples regardless of input
- Solutions:
  - Minibatch discrimination
  - Feature matching
  - Unrolled discriminator
  - Spectral normalization

### 2. Training Instability
- Oscillating losses
- Divergence
- Solutions:
  - Careful learning rate tuning
  - Batch normalization
  - Gradient penalty (WGAN-GP)
  - Spectral normalization

### 3. Vanishing Gradients
- When discriminator is too good
- Generator gets no useful gradient signal
- Solutions:
  - Wasserstein loss
  - Spectral normalization
  - Progressive growing

### 4. Computational Cost
- Requires training two networks
- Training iterations can be large
- GPU memory intensive

## Advanced GAN Variants

### Conditional GAN (cGAN)
- Condition generation on additional information
- G(z|c), D(x|c)
- Applications: Image-to-image translation, guided generation

### Spectral Normalization GAN
- Normalizes discriminator weights
- Ensures 1-Lipschitz constraint
- Improved training stability

### Progressive GAN
- Start with low resolution
- Gradually add layers for higher resolution
- Better high-quality image generation
- More stable training

### StyleGAN
- Adaptive instance normalization (AdaIN)
- Style control at different layers
- Disentangled representations
- State-of-the-art quality

## Implementation Details

### Hyperparameters
- Learning rate: typically 0.0002
- Beta1 (Adam): typically 0.5
- Batch size: 32-128
- Training iterations: 100k-1M

### Architecture Tips
- Use LeakyReLU in discriminator (not ReLU)
- Use batch normalization (skip in generator first/last layer)
- Use transposed convolutions in generator
- Avoid pooling, use strided convolutions

### Training Tricks
- Label smoothing: use 0.9 instead of 1.0 for real labels
- One-sided label smoothing
- Gradient penalty: λ * |∇_x D(x)|²
- Train discriminator more often than generator

## Evaluation Metrics

### Inception Score (IS)
- Quality and diversity of generated samples
- Higher is better
- IS = exp(E[KL(p(y|x)||p(y))])

### Fréchet Inception Distance (FID)
- Distance between real and fake distributions
- Lower is better
- More robust than IS
- Standard metric in current literature

### Human Evaluation
- Most reliable
- Resource-intensive
- Subjective quality assessment

## Applications

1. **Image Generation**
   - High-resolution face synthesis
   - Scene generation
   - Object generation

2. **Image-to-Image Translation**
   - Pix2Pix, CycleGAN
   - Domain transfer
   - Style transfer

3. **Super-resolution**
   - SRGAN, ESRGAN
   - Upsampling images
   - Enhancing details

4. **Data Augmentation**
   - Generate additional training samples
   - Improve model robustness

5. **Anomaly Detection**
   - Deviation from learned distribution
   - Medical imaging applications

## Common Pitfalls

1. **Not normalizing inputs** - Normalize to [-1, 1] or [0, 1]
2. **Using ReLU in discriminator** - Use LeakyReLU
3. **Training generator and discriminator equally** - Usually train D more
4. **Ignoring batch normalization** - Critical for stability
5. **Not using gradient penalty** - Important for WGAN

## Resources

- GAN Original Paper: Goodfellow et al., 2014
- Spectral Normalization: Miyato et al., 2018
- Progressive GAN: Karras et al., 2017
- StyleGAN: Karras et al., 2019
