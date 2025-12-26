#!/usr/bin/env python3
"""
Hello Machine Learning - A simple introduction to ML with Python

This script demonstrates basic ML concepts:
1. Loading data
2. Creating a simple model
3. Making predictions
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Hello Machine Learning ===")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading the Iris dataset...")
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target/Label
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print()
    
    # Step 2: Split data
    print("Step 2: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print()
    
    # Step 3: Create and train model
    print("Step 3: Training a Decision Tree classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    print()
    
    # Step 4: Make predictions
    print("Step 4: Making predictions...")
    y_pred = model.predict(X_test)
    print(f"Sample predictions: {y_pred[:5]}")
    print(f"Sample actual values: {y_test[:5]}")
    print()
    
    # Step 5: Evaluate model
    print("Step 5: Evaluating the model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Step 6: Make a single prediction
    print("Step 6: Making a prediction for a new sample...")
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(new_sample)
    prediction_proba = model.predict_proba(new_sample)
    
    print(f"New sample features: {new_sample[0]}")
    print(f"Predicted class: {iris.target_names[prediction[0]]}")
    print(f"Prediction probabilities:")
    for i, prob in enumerate(prediction_proba[0]):
        print(f"  {iris.target_names[i]}: {prob:.2%}")
    print()
    print("=== Machine Learning is Fun! ===")

if __name__ == "__main__":
    main()
