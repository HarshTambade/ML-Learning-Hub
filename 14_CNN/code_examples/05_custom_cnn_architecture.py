"""Custom CNN Architectures
Implement custom CNN designs for specific tasks
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + identity
        return torch.relu(x)

class InceptionBlock(nn.Module):
    """Inception block with parallel convolutions"""
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionBlock, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)

class CustomResNet(nn.Module):
    """Custom ResNet-like architecture with residual blocks"""
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomInceptionNet(nn.Module):
    """Custom Inception-like architecture"""
    def __init__(self, num_classes=10):
        super(CustomInceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.inception1 = InceptionBlock(32, 16, 16, 32, 8, 16, 8)
        self.inception2 = InceptionBlock(72, 32, 32, 64, 16, 32, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(144, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.inception1(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.inception2(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DenseBlock(nn.Module):
    """Dense block with feature reuse"""
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=1)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return torch.cat([x, out], 1)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test custom architectures
    resnet = CustomResNet(num_classes=10).to(device)
    inception_net = CustomInceptionNet(num_classes=10).to(device)
    
    x = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        out1 = resnet(x)
        out2 = inception_net(x)
    
    print(f'ResNet output shape: {out1.shape}')
    print(f'InceptionNet output shape: {out2.shape}')
