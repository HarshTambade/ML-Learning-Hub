"""Exercise 1: Build and train a basic autoencoder

TODO:
1. Implement Encoder class (input -> latent dimension)
2. Implement Decoder class (latent -> input)
3. Combine into Autoencoder class
4. Write training loop with MSE loss
5. Visualize original vs reconstructed images
6. Test on MNIST dataset

Hints:
- Use torch.nn.Sequential for simplicity
- Input size: 28x28=784, latent: 64
- Use ReLU for hidden layers, Sigmoid for output
- Learning rate: 1e-3, epochs: 20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO: Implement your autoencoder here
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # TODO: Complete implementation
        pass

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # TODO: Complete implementation
        pass

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

if __name__ == "__main__":
    # TODO: Write training code
    # Load MNIST
    # Create model
    # Train for 20 epochs
    # Plot results
    pass
