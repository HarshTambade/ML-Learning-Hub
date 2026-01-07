# MNIST Digit Recognition Project

## Project Overview

Build a neural network to classify handwritten digits from the MNIST dataset with high accuracy (>95%).

## Dataset

- **MNIST**: 70,000 images of handwritten digits (0-9)
- 60,000 training samples
- 10,000 test samples
- 28x28 pixel grayscale images
- 10 classes (digits 0-9)

## Project Goals

1. Load and explore the MNIST dataset
2. Preprocess images (normalization, flattening)
3. Design a neural network architecture
4. Train the model with different hyperparameters
5. Evaluate on test set
6. Visualize predictions and error cases
7. Achieve >95% test accuracy

## Network Architecture

### Basic Approach (Dense Network)
- Input layer: 784 neurons (28*28)
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU  
- Output layer: 10 neurons, Softmax
- Loss: Categorical Cross-Entropy

### Advanced Approach (CNN)
- Conv2D: 32 filters (3x3)
- Conv2D: 64 filters (3x3)
- MaxPooling2D (2x2)
- Flatten
- Dense: 128 neurons, ReLU
- Dropout: 0.5
- Dense: 10 neurons, Softmax

## Implementation Steps

### Step 1: Load Data
```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Step 2: Preprocess
```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

### Step 3: Build Model
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

### Step 4: Train
```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, batch_size=128)
```

### Step 5: Evaluate
```python
accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
```

## Expected Results

- Dense network: 97-98% accuracy
- CNN network: 98-99% accuracy
- Training time: 1-5 minutes (CPU)

## Deliverables

1. Trained model file
2. Performance metrics (accuracy, loss curves)
3. Confusion matrix
4. Sample predictions visualization
5. Summary report with findings
