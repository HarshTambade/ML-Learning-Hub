"""
Normalizing Flows - Invertible Transformations for Generative Modeling

This module demonstrates normalizing flows - a powerful class of generative models
that use invertible transformations to map simple distributions to complex ones.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from typing import Tuple


class InvertibleBlock(nn.Module):
    """
    Basic invertible block using coupling layers.
    
    In coupling layers:
    - Half of dimensions are transformed deterministically
    - Other half act as conditioning variables
    - This ensures the transformation is invertible
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.split_dim = input_dim // 2
        
        # Transformation network for scale and translation
        self.transform_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim * 2)  # scale + translate
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (z -> x)
        
        Args:
            z: Latent tensor
            
        Returns:
            x: Transformed tensor
            log_det: Log determinant of Jacobian
        """
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
        
        # Get transformation parameters from first half
        params = self.transform_net(z1)
        scale, translate = params[:, :self.split_dim], params[:, self.split_dim:]
        
        # Apply affine transformation to second half
        x2 = z2 * torch.exp(scale) + translate
        x = torch.cat([z1, x2], dim=1)
        
        # Log determinant contribution
        log_det = scale.sum(dim=1)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass (x -> z)
        
        Args:
            x: Data tensor
            
        Returns:
            z: Latent tensor
            log_det: Log determinant of Jacobian (for inverse)
        """
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Get transformation parameters
        params = self.transform_net(x1)
        scale, translate = params[:, :self.split_dim], params[:, self.split_dim:]
        
        # Invert the transformation
        z2 = (x2 - translate) * torch.exp(-scale)
        z = torch.cat([x1, z2], dim=1)
        
        # Log determinant for inverse (negative of forward)
        log_det = -scale.sum(dim=1)
        
        return z, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow Model
    
    Stacks multiple invertible transformations to map from:
    simple distribution p(z) -> complex distribution p(x)
    
    Mathematical formulation:
    p(x) = p(z) * |det(dz/dx)|
    log p(x) = log p(z) - sum(log|det(Jacobian)|)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_flows: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_flows = num_flows
        
        # Stack of invertible blocks
        self.flows = nn.ModuleList([
            InvertibleBlock(input_dim, hidden_dim)
            for _ in range(num_flows)
        ])
        
        # Prior distribution (standard normal)
        self.register_buffer('prior_mean', torch.zeros(input_dim))
        self.register_buffer('prior_scale', torch.ones(input_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log likelihood of data
        
        log p(x) = log p(z_k) + sum_i log|det(J_i)|
        
        Args:
            x: Data tensor
            
        Returns:
            log_likelihood: Log probability of data
            z: Final latent representation
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        # Apply each flow transformation
        for flow in self.flows:
            z, log_det = flow.inverse(z)
            log_det_total += log_det
        
        # Compute prior log likelihood
        prior_dist = Normal(self.prior_mean, self.prior_scale)
        log_prob_prior = prior_dist.log_prob(z).sum(dim=1)
        
        # Final log likelihood
        log_likelihood = log_prob_prior + log_det_total
        
        return log_likelihood, z
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated data
        """
        # Sample from prior
        prior_dist = Normal(self.prior_mean, self.prior_scale)
        z = prior_dist.rsample((num_samples,))
        
        # Apply forward transformations
        for flow in reversed(self.flows):
            z, _ = flow.forward(z)
        
        return z
    
    def compute_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute exact likelihood (not approximate like VAEs)
        
        Args:
            x: Data tensor
            
        Returns:
            likelihood: Probability density
        """
        log_likelihood, _ = self.forward(x)
        return torch.exp(log_likelihood)


def train_flow(model, train_loader, num_epochs=10, learning_rate=1e-3):
    """
    Train normalizing flow model
    
    Args:
        model: NormalizingFlow instance
        train_loader: DataLoader for training
        num_epochs: Number of epochs
        learning_rate: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            x = batch[0]
            
            # Forward pass
            log_likelihood, _ = model(x)
            
            # Loss: negative log likelihood
            loss = -log_likelihood.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Example usage
    print("Normalizing Flows Example")
    print("=" * 50)
    
    # Create model
    input_dim = 2
    model = NormalizingFlow(input_dim=input_dim, hidden_dim=128, num_flows=4)
    
    # Generate samples
    print("\nGenerating samples from prior...")
    samples = model.sample(num_samples=1000)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {samples.mean(dim=0)}")
    print(f"Sample std: {samples.std(dim=0)}")
    
    # Compute likelihood for random data
    print("\nComputing likelihood for random data...")
    test_data = torch.randn(10, input_dim)
    log_prob, z = model(test_data)
    print(f"Log probabilities: {log_prob}")
    print(f"Mean log probability: {log_prob.mean():.4f}")
    
    # Key advantages of normalizing flows:
    print("\nKey Advantages of Normalizing Flows:")
    print("1. Exact likelihood computation (not approximate like VAEs)")
    print("2. Invertible transformations allow sampling and inference")
    print("3. Flexible model class - can approximate any distribution")
    print("4. Good for density estimation and anomaly detection")
    print("\nKey Limitations:")
    print("1. Computational cost - need invertible transformations")
    print("2. Limited architectural flexibility")
    print("3. Harder to scale to very high dimensions")
