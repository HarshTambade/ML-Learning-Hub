"""PixelCNN - Autoregressive Image Generation"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedConv2d(nn.Module):
    """Masked convolution for autoregressive generation."""
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='A'):
        super().__init__()
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', torch.ones(1, 1, kernel_size, kernel_size))
        if mask_type == 'A':
            self.mask[:, :, kernel_size // 2:, :] = 0
            self.mask[:, :, kernel_size // 2, kernel_size // 2:] = 0
        elif mask_type == 'B':
            self.mask[:, :, kernel_size // 2:, :] = 0
            self.mask[:, :, kernel_size // 2, kernel_size // 2 + 1:] = 0
    
    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

class PixelCNN(nn.Module):
    """PixelCNN model for autoregressive generation."""
    def __init__(self, img_channels=3, num_colors=256, num_layers=7):
        super().__init__()
        self.img_channels = img_channels
        self.num_colors = num_colors
        
        layers = [MaskedConv2d(img_channels, 128, 7, 'A')]
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2d(128, 128, 3, 'B'))
        layers.extend([nn.ReLU(), nn.Conv2d(128, num_colors * img_channels, 1)])
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """P(x_i|x_1:i-1)"""
        return self.net(x)
    
    def sample(self, shape, device):
        """Generate samples pixel by pixel."""
        x = torch.zeros(shape, device=device)
        for i in range(shape[2]):
            for j in range(shape[3]):
                logits = self.forward(x)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                x[:, :, i, j] = torch.multinomial(probs, 1).squeeze(1)
        return x

if __name__ == "__main__":
    model = PixelCNN(img_channels=3)
    samples = model.sample((4, 3, 28, 28), torch.device('cpu'))
    print(f"Generated samples: {samples.shape}")
