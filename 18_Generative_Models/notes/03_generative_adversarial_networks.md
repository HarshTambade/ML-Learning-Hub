# Generative Adversarial Networks (GANs)

## Overview

Generative Adversarial Networks introduce a novel approach to generative modeling through adversarial training. Two neural networks compete: a generator creates fake data while a discriminator learns to distinguish real from fake.

### Key Characteristics

- **Adversarial Framework**: Generator vs Discriminator competition
- **Implicit Distribution Learning**: No explicit probability model
- **High-Quality Samples**: Excellent image generation quality
- **Minimax Game Theory**: Nash equilibrium at convergence
- **No Encoder**: Cannot infer latent codes from data

## Mathematical Foundation

### The GAN Objective

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Where:
- **G**: Generator network (creates fake samples)
- **D**: Discriminator network (classifies real vs fake)
- **x**: Real data samples
- **z**: Noise/latent vectors

### Game Theory Perspective

**Generator's Goal**: Minimize V(D, G)
- Wants to fool the discriminator
- Wants D(G(z)) → 1

**Discriminator's Goal**: Maximize V(D, G)
- Wants to correctly classify real and fake
- Wants D(x) → 1 and D(G(z)) → 0

### Nash Equilibrium

At convergence (theoretical):
```
G* generates from real data distribution
D* achieves 50% accuracy (random guessing)
```

## Architecture

### Generator Network

Maps latent code z to data space:

```
z (low-dim noise) → Dense → ReLU → Dense → BatchNorm → ReLU 
  → Deconv → Deconv → Tanh → fake image
```

**Key Design Choices:**
- Input: Random noise from simple distribution (uniform or normal)
- Hidden layers: Progressively increase spatial dimensions
- Output: Same shape as real data
- Activation: Typically tanh for [-1, 1] output range
- Normalization: BatchNorm helps stabilize training

### Discriminator Network

Classifies whether input is real or fake:

```
image → Conv → LeakyReLU → Conv → BatchNorm → LeakyReLU
  → Flatten → Dense → Sigmoid → [0, 1] probability
```

**Key Design Choices:**
- Input: Both real and generated images
- Hidden layers: Progressively reduce spatial dimensions
- Output: Single value (probability of being real)
- Activation: LeakyReLU prevents dead neurons
- Normalization: Spectral normalization for stability

## Training Dynamics

### Training Procedure

**Iteration t:**

1. **Discriminator Update (k times):**
   - Sample real batch x from data
   - Sample noise z
   - Generate fake x_fake = G(z)
   - Compute: L_D = -log D(x) - log(1 - D(x_fake))
   - Update D via gradient ascent (or -L_D gradient descent)

2. **Generator Update (once):**
   - Sample noise z
   - Generate x_fake = G(z)
   - Compute: L_G = -log D(G(z)) or L_G = log(1 - D(G(z)))
   - Update G via gradient descent

### Loss Functions Variants

**Standard GAN:**
```
L_G = -log D(G(z))  (can cause saturation)
L_D = -log D(x) - log(1 - D(G(z)))
```

**Improved: Wasserstein GAN (WGAN):**
```
L_G = -E[D(G(z))]
L_D = E[D(G(z))] - E[D(x)] + λ·GP

Where GP = gradient penalty to enforce 1-Lipschitz constraint
```

**Least Squares GAN:**
```
L_G = E[(D(G(z)) - 1)²]
L_D = E[(D(x) - 1)²] + E[(D(G(z)))²]
```

## Training Challenges & Solutions

### Mode Collapse

**Problem**: Generator produces limited diversity
- Falls into local mode
- Generates only certain types of samples
- Discriminator overfits to generated patterns

**Indicators:**
- Generated images look very similar
- Discriminator achieves 100% accuracy
- Diversity loss during training

**Solutions:**
- **Feature Matching**: Match feature statistics instead of raw pixels
- **Minibatch Discrimination**: Discriminator looks at differences in batch
- **Unrolled GANs**: Unroll discriminator gradient
- **Wasserstein Distance**: Better loss function
- **Spectral Normalization**: Lipschitz constant control

### Training Instability

**Problem**: Loss oscillates wildly or diverges

**Causes:**
- Imbalanced discriminator/generator strength
- Poor learning rate
- Insufficient normalization

**Solutions:**
- **Batch Normalization**: Normalize activations
- **Spectral Normalization**: Control discriminator Lipschitz
- **Gradient Penalty**: Enforce smoothness
- **Separate Learning Rates**: Often slower for D
- **Feature Matching**: Reduce overfitting

### Vanishing Gradients

**Problem**: Gradients become too small for updates

**Why**: When discriminator is too good, D(G(z)) ≈ 0, log(1-D(G(z))) saturates

**Solutions:**
- Use alternative loss: max log D(G(z)) instead of min log(1-D(G(z)))
- Wasserstein distance
- Relativistic discriminator

## Important Concepts

### Spectral Normalization

Normalize discriminator weights to have spectral norm = 1:

```
W_normalized = W / σ(W)

Where σ(W) = largest singular value
```

**Benefits:**
- Ensures 1-Lipschitz discriminator
- Stabilizes training significantly
- Allows higher learning rates

### Conditional GANs (cGAN)

Generate data conditioned on class label:

```
G(z | y) → generate fake of class y
D(x, y) → discriminate given class y
```

Useful for class-specific generation and image-to-image translation.

### Progressive GANs

Grow network layers during training:

```
Phase 1: 4×4 → 4×4
Phase 2: 4×4 → 8×8
Phase 3: 8×8 → 16×16
...
```

**Benefits:**
- Stability: Easier to train progressively
- Quality: Generate high-resolution images
- Speed: Start small, scale up

## GAN Variants

### DCGAN (Deep Convolutional GAN)
- Specific architecture guidelines
- Convolutional layers instead of fully connected
- BatchNorm in both G and D
- ReLU in generator, LeakyReLU in discriminator

### Wasserstein GAN
- Wasserstein distance instead of JS divergence
- Better gradient flow
- Fewer training tricks needed

### StyleGAN
- Style-based generation
- Adaptive instance normalization
- Incredible visual quality for faces

### CycleGAN
- Unpaired image-to-image translation
- Cycle consistency loss
- No paired training data needed

### Pix2Pix
- Paired image translation
- Conditional GAN with L1 loss
- Applications: sketch→photo, day→night

## Performance Metrics

### Inception Score (IS)

```
IS = exp(E_x[KL(P(y|x) || P(y))])
```

- Measures sample quality and diversity
- Higher is better (typical range: 3-10)
- Biased towards ImageNet-like features

### Frechet Inception Distance (FID)

```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real Σ_fake))
```

- Distance between real and generated feature distributions
- Lower is better
- More reliable than IS

### Precision & Recall

- Precision: % of generated samples on real data manifold
- Recall: % of real data manifold covered by generation
- Trade-off between quality and diversity

## Applications

### Image Generation
- Face synthesis
- Artistic style transfer
- Image inpainting and restoration
- Super-resolution

### Image Translation
- CycleGAN: unpaired translation
- Pix2Pix: paired translation
- Domain adaptation

### Data Augmentation
- Generate synthetic training data
- Improve downstream classifier
- Address class imbalance

### Text-to-Image
- Generate images from descriptions
- StackGAN: progressive refinement
- AttnGAN: attention-based generation

## Implementation Considerations

### Hyperparameters
- Learning rate: 0.0002 (common for Adam)
- Beta1: 0.5 (for Adam optimizer)
- Batch size: 32-128
- k (discriminator updates): 1-5 per generator update

### Architecture Guidelines
- Use Conv/Deconv instead of FC layers
- Apply BatchNorm except discriminator output
- Use LeakyReLU with slope 0.2
- Avoid pooling, use strides for downsampling

### Training Tips
- Start with spectral normalization
- Use gradient penalty for WGAN
- Monitor discriminator/generator loss ratio
- Visualize generated samples regularly
- Use multiple evaluation metrics

## Mathematical Deep Dive: JS Divergence vs Wasserstein

**Jensen-Shannon Divergence:**
```
JS(P || Q) = 1/2 KL(P || M) + 1/2 KL(Q || M)
M = (P + Q) / 2
```

- Symmetric unlike KL
- Can be 0 even when distributions don't overlap
- Causes vanishing gradients

**Wasserstein Distance:**
```
W(P, Q) = inf E[||X - Y||]
          where X ~ P, Y ~ Q
```

- Measures transport cost between distributions
- Meaningful gradient even when supports don't overlap
- Enables better training dynamics

## Conclusion

GANs represent a paradigm shift in generative modeling through:
- Adversarial training framework
- Implicit distribution learning
- Superior sample quality

Key takeaways:
- Training is inherently unstable (minimax formulation)
- Multiple tricks needed for practical training
- Variants address specific limitations
- Excellent for high-quality image synthesis
- Active research area with continuous improvements
