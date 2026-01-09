"""
Basic LSTM for Text Classification
Demonstrates sentiment analysis using LSTM on movie reviews
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict

# Sample data for demonstration
training_data = [
    ("this movie is great", 1),
    ("terrible film", 0),
    ("absolutely wonderful", 1),
    ("waste of time", 0),
    ("amazing performance", 1),
    ("very disappointing", 0),
]

class TextClassificationDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, vocab=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab or self.build_vocab(texts)
        self.text_to_indices = [self.text_to_indices_list(t) for t in texts]
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def text_to_indices_list(self, text):
        """Convert text to indices"""
        indices = []
        for word in text.lower().split():
            idx = self.vocab.get(word, self.vocab["<UNK>"])
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.text_to_indices[idx], self.labels[idx]

def collate_batch(batch):
    """Pad sequences to same length"""
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels

class LSTMTextClassifier(nn.Module):
    """LSTM-based text classifier"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, text, text_lengths=None):
        """Forward pass"""
        embedded = self.embedding(text)  # (batch, seq_len, embed_dim)
        lstm_output, (hidden, cell) = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        
        # Use last hidden state for classification
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, num_classes)
        
        return out

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    texts, labels = zip(*training_data)
    dataset = TextClassificationDataset(texts, labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_batch)
    
    # Initialize model
    vocab_size = len(dataset.vocab)
    model = LSTMTextClassifier(
        vocab_size=vocab_size,
        embed_dim=100,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 10
    
    print(f"Training on {device}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    print("\nTraining complete!")
