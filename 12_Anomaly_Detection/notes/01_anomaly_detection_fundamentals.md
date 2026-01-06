# Anomaly Detection Fundamentals

## Table of Contents
1. [What is Anomaly Detection?](#what-is-anomaly-detection)
2. [Key Characteristics](#key-characteristics)
3. [Types of Anomalies](#types-of-anomalies)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Challenges in Anomaly Detection](#challenges)
6. [Applications](#applications)
7. [Methods Overview](#methods-overview)
8. [Resources](#resources)

---

## What is Anomaly Detection?

Anomaly detection is a machine learning technique used to identify observations that deviate significantly from the majority of data. It focuses on finding unusual patterns, outliers, and novelties that differ from normal behavior.

### Formal Definition

Given a dataset D = {x1, x2, ..., xn}, an observation xi is considered an anomaly if it significantly deviates from the expected normal pattern. Mathematically:

```
Anomaly: P(x | normal) << P(x | anomaly)
```

### Why is it Important?

- **Early Detection**: Identify problems before they escalate
- **Cost Reduction**: Prevent losses from fraud, equipment failure, etc.
- **Safety**: Detect security threats and dangerous behaviors
- **Quality Assurance**: Maintain product and service standards

---

## Key Characteristics

### 1. **Unsupervised Learning**
- Often deals with unlabeled data
- No predefined "normal" or "anomalous" labels
- Model must learn patterns from mostly normal data

### 2. **No Gold Standard**
- Hard to define what is "normal" in complex systems
- Different contexts may have different definitions
- Subjective nature of anomalies

### 3. **Imbalanced Data**
- Anomalies are typically rare (< 1% of data)
- Class imbalance makes detection challenging
- Standard metrics (accuracy) become misleading

### 4. **Real-world Complexity**
- Normal behavior changes over time (concept drift)
- Seasonal patterns affect what's "normal"
- Different data distributions across time periods

### 5. **High Dimensionality**
- Real-world data often has hundreds or thousands of features
- Curse of dimensionality affects detection accuracy
- Feature selection becomes critical

---

## Types of Anomalies

### 1. Point Anomalies (Global Outliers)

Individual data points that deviate significantly from normal patterns.

**Characteristics**:
- Single instance is anomalous
- Stands out in the entire dataset
- Most common type of anomaly

**Example**: 
```
Credit card transaction of $50,000 when typical transactions are $50-200
Fever temperature of 105°F when normal is 98.6°F
```

**Visual Representation**:
```
   Normal data (o), Anomalies (X)
   
   o  o  o  o  o  o
   o     o  X  o  o  o
   o  o  o  o  o
      o  o  o  o  X  o
   o  o  o  o  o  o
```

### 2. Contextual Anomalies (Conditional Outliers)

Data points that are anomalous in a specific context but normal otherwise.

**Characteristics**:
- Normal value, but unusual context
- Context-dependent definition of anomaly
- Requires domain knowledge

**Examples**:
```
Temperature of 90°F is normal in summer but anomalous in winter
Network traffic spike at 3 AM (unusual) vs 2 PM (normal)
High CPU usage during batch processing (normal) vs regular usage (anomalous)
```

**Visual Representation**:
```
Context 1 (Winter):     Context 2 (Summer):
   o  o  o  X  o           X  X  X  X  X
   o  o  o  o  o           X  X  X  X  X  (90°F is normal here)
   (90°F is anomalous)     X  X  X  X  X
```

### 3. Collective Anomalies

A group of data points that together form an anomalous pattern, though individual points may be normal.

**Characteristics**:
- Individual points are normal
- Pattern formed is unusual
- Interdependencies matter

**Examples**:
```
Gradual network traffic increase over hours (each increment normal, pattern anomalous)
Price decrease across all products simultaneously (individual prices normal, pattern suspicious)
Sequence of small transactions to avoid detection threshold (individual transactions normal)
```

**Visual Representation**:
```
Time Series Data:
    |
    |      * * *     <- Collective anomaly
    |    * * * * *
    |  * * * * * *
    |* * * * * * *
    |_______________
    (Individual points within normal range, but pattern is unusual)
```

---

## Mathematical Foundations

### 1. Statistical Approach

**Z-Score Method**:
```
z = (x - μ) / σ
```
Where:
- x = data point
- μ = mean
- σ = standard deviation

If |z| > 3, point is typically considered anomalous (99.7% of normal data within 3σ)

**Example**:
```python
import numpy as np
from scipy import stats

data = [10, 12, 11, 10, 11, 50, 12, 11]  # 50 is anomaly
z_scores = np.abs(stats.zscore(data))
anomalies = np.where(z_scores > 3)[0]
print(f"Anomalies at indices: {anomalies}")  # Output: [5]
```

### 2. Distance-Based Approach

**Euclidean Distance**:
```
d(x1, x2) = √(Σ(xi - yi)²)
```

Points with large distances from neighbors are anomalies.

**K-NN Method**:
- Find k nearest neighbors
- Calculate average distance to neighbors
- If distance > threshold, it's an anomaly

### 3. Density-Based Approach

**Local Outlier Factor (LOF)**:
```
LOF(x) = (average density of neighbors) / (density of x)
```

If LOF >> 1, point is in lower density region (anomaly)

### 4. Information Theory

**Entropy-Based**:
```
AnomalyScore = -log(P(x|θ))
```

Lower probability = higher anomaly score

---

## Challenges in Anomaly Detection

### 1. **Lack of Labels**
- Most data is unlabeled
- Difficult to train supervised models
- Can't use standard supervised learning metrics
- Solution: Semi-supervised or unsupervised methods

### 2. **Rarity of Anomalies**
- Makes model training difficult
- Class imbalance problem
- Standard metrics (accuracy) become meaningless
- Need: Precision, Recall, F1-Score, AUC-ROC

### 3. **Evolving Patterns**
- Normal behavior changes over time (concept drift)
- What's normal in January may not be normal in July
- Requires adaptive/online learning
- Example: Seasonal effects in sales data

### 4. **High Dimensionality**
- Many features make detection harder
- Curse of dimensionality
- Distances become less meaningful in high dimensions
- Solution: Feature selection/dimensionality reduction

### 5. **Domain Expertise Required**
- Understanding domain crucial for defining "normal"
- Parameter tuning varies by domain
- Thresholds differ across applications
- Requires collaboration with domain experts

### 6. **False Positives vs False Negatives**
- Different costs in different domains
- Fraud detection: False positives annoying, false negatives costly
- Medical diagnosis: False negatives dangerous, false positives expensive

---

## Applications

### Finance
- **Credit Card Fraud Detection**: Unusual spending patterns
- **Money Laundering Detection**: Suspicious transaction sequences
- **Unusual Trading Patterns**: Insider trading detection
- **Credit Risk Assessment**: Abnormal behavior predicting defaults

**Industry Impact**: Fraud costs financial institutions billions annually

### Cybersecurity
- **Network Intrusion Detection**: Unauthorized access attempts
- **Malware Detection**: Unusual system behavior
- **Unauthorized Access Attempts**: Failed login patterns
- **DDoS Attack Detection**: Abnormal traffic patterns

**Example**: 
```
Normal: 100 requests/second from various IPs
Anomaly: 10,000 requests/second from single IP
```

### Manufacturing
- **Equipment Failure Prediction**: Abnormal vibration/temperature
- **Quality Control**: Defective products detection
- **Predictive Maintenance**: Early warning signs of equipment failure
- **Production Line Monitoring**: Deviation from specifications

**Business Value**: Reducing downtime and maintenance costs

### Healthcare
- **Disease Detection**: Abnormal vital signs or lab results
- **Patient Monitoring**: Unusual health metrics
- **Drug Safety**: Adverse reaction detection
- **Medical Fraud**: Unusual billing patterns

### E-commerce
- **Bot Detection**: Unusual user behavior (clicks, purchases)
- **Return Fraud**: Suspicious return patterns
- **Price Anomalies**: Incorrect pricing
- **Inventory Anomalies**: Stock discrepancies

### IoT & Sensor Networks
- **Sensor Failure Detection**: Malfunctioning sensors
- **Environmental Monitoring**: Unusual readings
- **Energy Consumption**: Abnormal usage patterns

---

## Methods Overview

### 1. Statistical Methods
- Z-Score
- Isolation Forest
- Gaussian Mixture Models (GMM)

### 2. Distance-Based Methods
- K-Nearest Neighbors (K-NN)
- Local Outlier Factor (LOF)
- DBSCAN

### 3. Density-Based Methods
- Kernel Density Estimation (KDE)
- One-Class SVM
- Elliptic Envelope

### 4. Reconstruction-Based Methods
- Principal Component Analysis (PCA)
- Autoencoders
- Variational Autoencoders (VAE)

### 5. Ensemble Methods
- Isolation Forest
- Robust Random Cut Forest

### 6. Deep Learning Methods
- LSTM-based approaches
- Convolutional Autoencoders
- Generative Adversarial Networks (GANs)

---

## Resources

### Books
1. "Outlier Detection for Temporal Data" by Gupta et al.
2. "Anomaly Detection Principles and Algorithms" by Tan et al.
3. "Machine Learning" by Tom Mitchell - Chapter on Outlier Detection

### Online Courses
- Coursera: Machine Learning by Andrew Ng
- Udacity: Machine Learning Nanodegree
- Fast.ai: Practical Deep Learning for Coders

### Libraries & Tools
- **Python Libraries**:
  - scikit-learn: sklearn.covariance, sklearn.ensemble
  - PyOD: Comprehensive Python toolkit for outlier detection
  - TensorFlow/PyTorch: Deep learning methods
  - statsmodels: Statistical methods

- **Cloud Services**:
  - AWS: SageMaker
  - Azure: Anomaly Detector
  - Google Cloud: AI Platform

### Research Papers
- "Isolation Forest" by Liu et al. (2008)
- "Local Outlier Factor" by Breunig et al. (2000)
- "Deep Autoencoder Neural Networks" (2015 onwards)

### Datasets for Practice
- **UCI Machine Learning Repository**: Multiple anomaly datasets
- **KDD Cup 1999**: Network intrusion detection
- **UNSW-NB15**: Cyber attack dataset
- **Credit Card Fraud Detection**: Kaggle
- **ECG5000**: ECG anomalies
- **Thyroid Disease**: Medical anomalies

### Useful Links
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Scikit-learn Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Kaggle Anomaly Detection Competitions](https://www.kaggle.com/)
- [Research Papers on ArXiv](https://arxiv.org/list/cs.LG/recent)

---

## Next Steps

Explore the following topics:
1. **02_statistical_methods.md** - Statistical approaches (Z-Score, IQR, Mahalanobis)
2. **03_isolation_forest_methods.md** - Tree-based anomaly detection
3. **04_local_outlier_factor.md** - Density-based methods
4. **05_deep_learning_approaches.md** - Neural network-based methods

---

**Last Updated**: January 2026
**Difficulty Level**: Beginner to Intermediate
**Prerequisites**: Basic Python, Statistics, Machine Learning fundamentals
