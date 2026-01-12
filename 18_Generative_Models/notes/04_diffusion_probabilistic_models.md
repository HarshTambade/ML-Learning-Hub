# Diffusion Probabilistic Models - Comprehensive Deep Dive

## Table of Contents
1. Introduction and Intuition
2. Forward Diffusion Process
3. Reverse Diffusion Process
4. Mathematical Formulation
5. Network Architecture (U-Net)
6. Training Process
7. Sampling and Generation
8. Noise Scheduling Strategies
9. Practical Implementation
10. Advanced Techniques
11. Applications
12. Comparison with Other Models

## Part 1: Introduction and Intuition

### Core Idea
Diffusion models generate data by gradually removing noise from random input:
- **Forward Process**: Add noise to data until it becomes pure noise
- **Reverse Process**: Remove noise step-by-step to generate new samples

### Why Diffusion Models?
- State-of-the-art image quality (SOTA in many benchmarks)
- Stable training compared to GANs
- Tractable likelihood computation
- Strong theoretical foundations

### High-Level Process

```
Original Image
    |
    v (Add Noise)
Noisy Image Step 1
    |
    v (Add Noise)
Noisy Image Step 2
    |
    ...
    |
    v (Add Noise)
Pure Noise (Step T)
    |
    ^ (Remove Noise - Learned)
Noisy Image Step T-1
    |
    ^ (Remove Noise - Learned)
Noisy Image Step T-2
    |
    ...
    |
    ^ (Remove Noise - Learned)
Generated Image (Clean)
```

## Part 2: Forward Diffusion Process

### Mathematical Definition

Start with data x_0 ~ q(x_0), gradually add Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

Where:
- beta_t: Noise schedule (increases with t)
- x_t: Noisy data at step t
- t ∈ [1, T]

### Noise Schedule

**Linear Schedule:**
```
beta_t = beta_start + (beta_end - beta_start) * (t / T)
beta_start = 0.0001
beta_end = 0.02
```

**Key Coefficients:**
```
alpha_t = 1 - beta_t
alpha_bar_t = ∏_{i=1}^{t} alpha_i  (Cumulative product)

# Direct sampling formula:
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
where epsilon ~ N(0, I)
```

### Properties

1. **Markovian**: q(x_t | x_{t-1})
2. **Tractable**: Can sample any x_t from x_0 directly
3. **Progressive**: Noise increases monotonically

## Part 3: Reverse Diffusion Process

### Reverse Process

```
p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t * I)
```

Where:
- mu_theta: Neural network predicting mean
- sigma_t: Variance (fixed during training)
- theta: Network parameters to learn

### Training Objective

Minimize KL divergence between true reverse and learned reverse:

```
L = KL(q(x_{t-1}|x_t, x_0) || p(x_{t-1}|x_t))
```

**Simplified Objective** (Equivalent):
```
L_simple = E_t[ ||epsilon - epsilon_theta(x_t, t)||^2 ]
```

Where:
- epsilon: True noise
- epsilon_theta: Network predicting noise
- Much simpler to implement and optimize

## Part 4: Mathematical Formulation

### Complete ELBO

```
log p(x_0) >= E_q[ log p(x_T) - sum_{t=1}^{T} KL(q(x_t|x_{t-1})||p(x_t|x_{t-1})) ]
```

Breakdown:
1. **Prior Matching**: log p(x_T) (x_T close to N(0,I))
2. **Denoising**: KL terms for each step
3. **Data**: How well we reconstruct from x_1

### Loss Functions

**Noise Prediction Loss:**
```
L_noise = E_{t,x_0,epsilon}[ ||epsilon - epsilon_theta(x_t, t)||^2 ]
where x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
```

**Score Matching:**
```
L_score = E_{t,x_0,epsilon}[ ||s_theta(x_t, t) - nabla_{x_t} log q(x_t|x_0)||^2 ]
```

**Variants:**
- Hybrid loss combining multiple objectives
- Weighted loss by time-step importance
- SNR (Signal-to-Noise Ratio) weighting

## Part 5: Network Architecture - U-Net

### Standard U-Net for Diffusion

```
Input: (B, 3, H, W) + timestep t
  |
  v
[Conv Blocks with Time Embedding]
  |
  +---> Encoder (Progressive Downsampling)
  |       |
  |       v
  |     Bottleneck Blocks
  |       |
  |       v
  |     Decoder (Progressive Upsampling)
  |       |
  +---> Skip Connections
  |
  v
Output: (B, 3, H, W)
```

### Key Components

**1. Time Embedding:**
```python
t_emb = TimeEmbedding(timestep)
# Typically: sin/cos positional encoding
# Output shape: (B, embedding_dim)
```

**2. Residual Blocks with Time:**
```python
def ResBlock(x, t_emb):
    h = Conv(x)
    h = h + TimeProjection(t_emb)
    h = ReLU(h)
    h = Conv(h)
    return h + x  # Skip connection
```

**3. Attention Blocks:**
```python
# Multi-head self-attention
# Applied at multiple scales
# Improves consistency and quality
```

**4. Channel Multiplier:**
```
Base channels: 64
Multiplier: [1, 2, 3, 4]
Channels at each level: [64, 128, 192, 256]
```

## Part 6: Training Process

### Algorithm

```
for epoch in range(num_epochs):
    for x_0 in dataset:
        # 1. Sample random time
        t ~ Uniform(1, T)
        
        # 2. Sample noise
        epsilon ~ N(0, I)
        
        # 3. Noisy sample
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        
        # 4. Forward pass
        epsilon_pred = model(x_t, t)
        
        # 5. Compute loss
        loss = MSE(epsilon_pred, epsilon)
        
        # 6. Backward pass
        loss.backward()
        optimizer.step()
```

### Training Tips

1. **Timestep Sampling**: Uniform vs weighted
2. **Loss Weighting**: SNR-based weighting important
3. **Batch Size**: Larger batches help convergence
4. **Learning Rate**: Start low (1e-4 to 1e-3)
5. **EMA**: Exponential moving average of weights

## Part 7: Sampling and Generation

### Ancestral Sampling

```python
def sample(model, shape, num_steps):
    x = randn(shape)  # Start from noise
    
    for t in range(num_steps-1, 0, -1):
        # Predict noise
        noise_pred = model(x, t)
        
        # Calculate reverse step
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_prev = alphas_cumprod[t-1]
        
        # Mean prediction
        x_0_pred = (x - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
        mean = (sqrt(alpha_bar_t_prev) * (1-alpha_t)/(1-alpha_bar_t)) * x_0_pred
        mean += (sqrt(alpha_t) * (1-alpha_bar_t_prev)/(1-alpha_bar_t)) * x
        
        # Add noise
        if t > 1:
            noise = randn_like(x)
            x = mean + sqrt(beta_t) * noise
        else:
            x = mean
    
    return x
```

### Efficient Sampling (DDIM)

```python
# Skip steps for faster generation
steps_to_skip = 10
selected_steps = range(0, num_steps, steps_to_skip)

# Use DDIM formula instead of ancestral sampling
# Result: 10-50x faster with minimal quality loss
```

## Part 8: Noise Scheduling Strategies

### Linear Schedule
```
beta_t = beta_start + (beta_end - beta_start) * t/T
```
**Pros**: Simple
**Cons**: Suboptimal for image generation

### Cosine Schedule
```
beta_t = sin(pi * (t+1)/(T+1))^2 - sin(pi * t/(T+1))^2
```
**Pros**: Better performance
**Cons**: More complex

### sqrt Schedule
```
beta_t = sqrt(alpha_bar_start) - sqrt(alpha_bar_end) * t/T
```
**Pros**: Gradual progression
**Cons**: Problem-dependent

## Part 9: Practical Implementation Tips

### Memory Optimization
1. Use 16-bit precision (fp16)
2. Gradient checkpointing
3. Smaller batch size with accumulation
4. Channel multipliers at different scales

### Speed Optimization
1. DDIM sampling (10-50x faster)
2. Progressive distillation
3. Latent diffusion (diffuse in latent space)
4. GPU optimization

## Part 10: Advanced Techniques

### 1. Conditional Diffusion
```
p(x_{t-1}|x_t, c) where c is condition (class, text, image)
```

### 2. Classifier-Free Guidance
```
epsilon_pred = epsilon_uncond + scale * (epsilon_cond - epsilon_uncond)
```

### 3. Inpainting
```
# Constrain known regions while diffusing unknown
for step in reverse_steps:
    x_t[known_mask] = known_values_noised
```

### 4. Image-to-Image
```
# Start diffusion from noisy version of input
# Less noise = more similar to input
```

## Part 11: Applications

1. **Image Generation**: High-quality synthesis
2. **Image Editing**: Inpainting, style transfer
3. **Super-resolution**: Upsampling with detail
4. **Video Generation**: Temporal consistency
5. **3D Generation**: 3D model synthesis
6. **Point Cloud**: 3D data generation

## Part 12: Comparison with Other Models

### Diffusion vs GAN
| Aspect | Diffusion | GAN |
|--------|-----------|-----|
| Training | Stable | Unstable |
| Likelihood | Approx | None |
| Sample Quality | Excellent | Excellent |
| Speed | Slow | Fast |
| Interpolation | Smooth | Rough |

### Diffusion vs VAE
| Aspect | Diffusion | VAE |
|--------|-----------|-----|
| Quality | SOTA | Good |
| Likelihood | Better | Tractable |
| Training | Easier | Stable |
| Flexibility | High | Medium |

## Summary

1. **Diffusion**: Gradually add noise then learn to reverse
2. **Training**: Predict noise at random steps
3. **Sampling**: Iteratively denoise from pure noise
4. **State-of-the-art**: Best image generation results
5. **Flexible**: Works for many tasks (inpaint, edit, etc.)
6. **Stable**: Much easier to train than GANs

## References
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)
- Classifier-Free Diffusion Guidance (Ho & Salimans, 2021)
- High-Resolution Image Synthesis with Latent Diffusion (Rombach et al., 2021)
