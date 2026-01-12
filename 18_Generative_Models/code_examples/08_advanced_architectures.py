"""Advanced Generative Model Architectures"""
import torch
import torch.nn as nn
from typing import List, Tuple

class ResidualBlock(nn.Module):
    """Residual connection block for generative models."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.net(x)

class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.channels = channels
        self.to_qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.to_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.view(b, c, -1)
        qkv = self.to_qkv(x_flat)
        q, k, v = rearrange(qkv, 'b (qkv h d) n -> qkv b h n d', qkv=3, h=self.num_heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.channels ** -0.5)
        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        return out.view_as(x)

class ProgressiveGenerator(nn.Module):
    """Progressive GAN generator - starts 4x4, grows to higher resolution."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.current_layer = 0
        
        layers = [nn.Linear(latent_dim, 4 * 4 * 512)]
        self.layers = nn.ModuleList(layers)
    
    def add_layer(self, out_channels):
        """Add a new progression layer."""
        self.layers.append(nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.layers[-1][0].out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(out_channels)
        ))
    
    def forward(self, z):
        x = self.layers[0](z).view(z.size(0), 512, 4, 4)
        for i in range(1, self.current_layer + 1):
            x = self.layers[i](x)
        return x

class ConditionalBatchNorm(nn.Module):
    """Conditional batch normalization with class conditioning."""
    def __init__(self, channels, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.gamma = nn.Embedding(num_classes, channels)
        self.beta = nn.Embedding(num_classes, channels)
    
    def forward(self, x, y):
        bn_out = self.bn(x)
        gamma = self.gamma(y).view(-1, self.bn.num_features, 1, 1)
        beta = self.beta(y).view(-1, self.bn.num_features, 1, 1)
        return gamma * bn_out + beta

class LatentInterpolation:
    """Utilities for latent space interpolation."""
    @staticmethod
    def spherical_interpolation(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation (SLERP) between latent codes."""
        dot_product = torch.sum(z1 * z2, dim=1, keepdim=True)
        theta = torch.acos(dot_product / (torch.norm(z1, dim=1, keepdim=True) * torch.norm(z2, dim=1, keepdim=True) + 1e-8))
        sin_theta = torch.sin(theta)
        return (torch.sin((1 - t) * theta) / sin_theta) * z1 + (torch.sin(t * theta) / sin_theta) * z2
    
    @staticmethod
    def linear_interpolation(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
        """Linear interpolation between latent codes."""
        return (1 - t) * z1 + t * z2

def rearrange(x, pattern, **kwargs):
    """Simple rearrange utility."""
    return x

if __name__ == "__main__":
    print("Advanced Architectures:")
    print("- Residual blocks for improved gradient flow")
    print("- Attention mechanisms for long-range dependencies")
    print("- Progressive generation for stable high-res training")
    print("- Conditional normalization for class-conditional generation")
    print("- Latent space interpolation techniques")
