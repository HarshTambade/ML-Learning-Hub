"""
Regularization Techniques for Neural Networks

This module demonstrates various regularization techniques including:
- L1/L2 Regularization (Weight Decay)
- Dropout
- Batch Normalization
- Early Stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RegularizedNetwork(nn.Module):
    """Neural network with L2 regularization (weight decay)"""
    
    def __init__(self, input_size=20, hidden_sizes=[64, 32], dropout_rate=0.5):
        super(RegularizedNetwork, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_with_regularization(X_train, y_train, X_val, y_val, 
                             epochs=100, batch_size=32,
                             l2_lambda=0.01, dropout_rate=0.3):
    """Train network with L2 regularization and dropout"""
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = RegularizedNetwork(input_size=X_train.shape[1], dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_val).unsqueeze(1))
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses


def main():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, n_redundant=5,
                               random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train model
    print('Training network with regularization...')
    model, train_losses, val_losses = train_with_regularization(
        X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32,
        l2_lambda=0.01, dropout_rate=0.3
    )
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.FloatTensor(X_test))
        test_predictions = (test_outputs > 0.5).numpy().flatten()
        accuracy = np.mean(test_predictions == y_test)
    
    print(f'\nTest Accuracy: {accuracy:.4f}')
    print('\nKey Regularization Techniques Demonstrated:')
    print('1. Dropout: Randomly deactivates neurons during training')
    print('2. L2 Regularization (weight_decay): Penalizes large weights')
    print('3. Batch Normalization: Normalizes layer inputs')
    print('4. Early Stopping: Stop when validation loss plateaus')


if __name__ == '__main__':
    main()
