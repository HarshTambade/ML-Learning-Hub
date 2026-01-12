"""Project 3: Anomaly Detection using VAE and Autoencoder
Detect anomalous patterns in data using reconstruction error

Objectives:
- Train VAE on normal MNIST digits (0-8)
- Use VAE for anomaly detection
- Detect digit 9 as anomaly
- Implement threshold-based detection
- Evaluate with ROC curve and AUC

Deliverables:
- Trained VAE model
- Reconstruction error distribution
- ROC curve analysis
- Confusion matrix
- Performance metrics (precision, recall, F1)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

class AnomalyDetector:
    """Anomaly Detection using Reconstruction Error"""
    
    def __init__(self, normal_class=None):
        # TODO: Initialize VAE model
        # TODO: Set normal class for one-class detection
        pass
    
    def train_on_normal(self, train_loader, epochs=50):
        # TODO: Train VAE on normal data only
        pass
    
    def compute_reconstruction_error(self, data):
        # TODO: Compute MSE reconstruction error
        # error = ||data - reconstructed||
        pass
    
    def detect_anomalies(self, test_loader, threshold=None):
        # TODO: Classify as normal or anomaly
        # Anomaly if error > threshold
        pass
    
    def find_optimal_threshold(self, errors, labels):
        # TODO: Find threshold that maximizes F1 score
        pass
    
    def evaluate(self, test_loader, normal_labels, anomaly_labels):
        # TODO: Compute ROC curve
        # TODO: Compute AUC score
        # TODO: Compute precision, recall, F1
        pass
    
    def plot_results(self, errors, labels, threshold=None):
        # TODO: Plot reconstruction error distribution
        # TODO: Mark threshold
        # TODO: Show normal vs anomaly samples
        pass

def main():
    # TODO: Load MNIST
    # TODO: Filter to normal digits (0-8)
    # TODO: Create anomaly set (digit 9)
    # TODO: Train detector
    # TODO: Evaluate on test set with anomalies
    # TODO: Plot ROC curve and error distribution
    pass

if __name__ == "__main__":
    main()
