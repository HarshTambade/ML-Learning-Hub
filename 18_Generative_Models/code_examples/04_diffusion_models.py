import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NoiseScheduler:
    """Linear noise schedule for diffusion"""
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def get_index_from_list(self, vals, t, x_shape):
        """Extract coefficients at specified timesteps"""
        batch_size = x_shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion(self, x_0, t):
        """Add noise to image at timestep t"""
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_1m_alpha = self.get_index_from_list(self.sqrt_1m_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha * x_0 + sqrt_1m_alpha * noise, noise

class DiffusionNetwork(nn.Module):
    """U-Net like denoising network"""
    def __init__(self, channels=1, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.enc1 = self._block(channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = self._block(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1 = self._block(64, 64)
        
        # Final output
        self.final = nn.Conv2d(64, channels, 3, 1, 1)
    
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # Decoder
        dec2 = self.dec2(self.upconv2(bottleneck))
        dec1 = self.dec1(self.upconv1(dec2))
        
        # Final output
        return self.final(dec1)

def train_diffusion(num_epochs=10, num_steps=100):
    """Train diffusion model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize models
    scheduler = NoiseScheduler(num_steps=num_steps)
    model = DiffusionNetwork().to(device)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    scheduler.sqrt_alphas_cumprod = scheduler.sqrt_alphas_cumprod.to(device)
    scheduler.sqrt_1m_alphas_cumprod = scheduler.sqrt_1m_alphas_cumprod.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            batch_size = x.size(0)
            
            # Random timesteps
            t = torch.randint(0, num_steps, (batch_size,)).to(device)
            
            # Forward diffusion
            x_t, noise = scheduler.forward_diffusion(x, t)
            
            # Predict noise
            noise_pred = model(x_t, t)
            
            # Compute loss
            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model, scheduler

@torch.no_grad()
def sample_diffusion(model, scheduler, num_samples=16, num_steps=100):
    """Generate samples using reverse diffusion"""
    device = next(model.parameters()).device
    x = torch.randn(num_samples, 1, 28, 28).to(device)
    
    for t in range(num_steps - 1, 0, -1):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long).to(device)
        noise_pred = model(x, t_tensor)
        x = x - noise_pred
    
    return x

if __name__ == "__main__":
    model, scheduler = train_diffusion(num_epochs=2, num_steps=100)
    samples = sample_diffusion(model, scheduler, num_samples=16)
    print("Diffusion model training complete!")
