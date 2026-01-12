import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Generator(nn.Module):
    """Generator network for GAN"""
    def __init__(self, latent_dim=100, img_channels=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.fc(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def train_gan(num_epochs=200, batch_size=128, latent_dim=100):
    """Train the GAN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(train_loader):
            real_data = real_data.to(device)
            batch_size_current = real_data.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            d_real_output = discriminator(real_data)
            d_real_loss = criterion(d_real_output, real_labels)
            
            # Fake data
            z = torch.randn(batch_size_current, latent_dim).to(device)
            fake_data = generator(z)
            d_fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size_current, latent_dim).to(device)
            fake_data = generator(z)
            d_fake_output = discriminator(fake_data)
            g_loss = criterion(d_fake_output, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}')
    
    return generator, discriminator

if __name__ == "__main__":
    generator, discriminator = train_gan(num_epochs=200)
    print("\nGAN training complete!")
