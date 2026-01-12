"""Exercise 2: Implement a basic GAN
TODO: Generator, Discriminator, training loop, binary cross-entropy loss"""
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        # TODO: z -> 28x28 images
        pass
    
    def forward(self, z):
        # TODO: Generate fake images
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 28x28 images -> real/fake probability
        pass
    
    def forward(self, x):
        # TODO: Classify real vs fake
        pass

if __name__ == "__main__":
    # TODO: Train GAN for 20 epochs
    pass
