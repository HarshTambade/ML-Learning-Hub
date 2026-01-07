# Simple Neural Network Implementation from Scratch

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """Simple 2-layer neural network (no external libraries)"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, m):
        # Backpropagation
        dz2 = self.a2 - y  # For softmax + cross-entropy
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=100, batch_size=32):
        m = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                batch_size_actual = X_batch.shape[0]
                
                # Forward
                output = self.forward(X_batch)
                
                # Compute loss (cross-entropy)
                batch_loss = -np.sum(y_batch * np.log(output + 1e-8)) / batch_size_actual
                epoch_loss += batch_loss * batch_size_actual
                
                # Backward
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, batch_size_actual)
                
                # Update
                self.update_weights(dW1, db1, dW2, db2)
            
            epoch_loss /= m
            losses.append(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == np.argmax(y, axis=1)) / len(y)
        return accuracy


# Example usage with Iris dataset
if __name__ == '__main__':
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # One-hot encoding
    y_one_hot = np.eye(3)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train network
    nn = NeuralNetwork(input_size=4, hidden_size=16, output_size=3, learning_rate=0.1)
    losses = nn.train(X_train, y_train, epochs=100, batch_size=16)
    
    # Evaluate
    train_acc = nn.evaluate(X_train, y_train)
    test_acc = nn.evaluate(X_test, y_test)
    
    print(f'\nTrain Accuracy: {train_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
