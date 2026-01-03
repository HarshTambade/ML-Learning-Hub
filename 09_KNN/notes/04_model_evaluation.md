# KNN Model Evaluation and Optimization

## 1. Classification Metrics

### Accuracy
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Percentage of correct predictions
- Best for balanced datasets
- Can be misleading with imbalanced data

### Precision
- Formula: TP / (TP + FP)
- Measures true positive rate among predicted positives
- Important when false positives are costly
- High precision = few false alarms

### Recall (Sensitivity)
- Formula: TP / (TP + FN)
- Measures true positive rate among actual positives
- Important when false negatives are costly
- High recall = few missed cases

### F1-Score
- Formula: 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Balanced metric for imbalanced datasets
- Range: 0 to 1 (higher is better)

## 2. Regression Metrics

### Mean Squared Error (MSE)
- Formula: (1/n) * sum((yi - ŷi)^2)
- Penalizes large errors more
- Always positive
- Same units as squared target variable

### Root Mean Squared Error (RMSE)
- Formula: sqrt(MSE)
- Same units as target variable
- More interpretable than MSE
- Widely used in practice

### Mean Absolute Error (MAE)
- Formula: (1/n) * sum(|yi - ŷi|)
- Average of absolute errors
- Less sensitive to outliers than RMSE
- Same units as target variable

## 3. Cross-Validation Techniques

### K-Fold Cross-Validation
- Divide data into k folds
- Train on k-1 folds, test on 1 fold
- Repeat k times and average scores
- Provides more reliable estimates
- Reduces variance in performance estimate

### Stratified K-Fold
- Maintains class distribution in each fold
- Essential for imbalanced classification
- Prevents biased fold splits

### Leave-One-Out CV (LOOCV)
- Train on n-1 samples, test on 1
- Computationally expensive
- Very high variance reduction
- Better for small datasets

## 4. Hyperparameter Tuning

### Grid Search
- Tests all combinations of parameters
- Exhaustive but computationally expensive
- Finds global optimum within grid
- Use for small parameter spaces

### Random Search
- Samples random parameter combinations
- More efficient than grid search
- Works better for high dimensions
- May miss optimal combination

### Bayesian Optimization
- Uses probability model to guide search
- Sequentially evaluates promising regions
- Efficient for expensive evaluations
- Better than random and grid search

## 5. Common KNN Parameters to Tune

### Number of Neighbors (K)
- Small K: Low bias, high variance
- Large K: High bias, low variance
- Typical range: 3-20
- Best value depends on data distribution

### Distance Metric
- Euclidean for numerical data
- Manhattan for features with outliers
- Cosine for text/high-dimensional data

### Weights
- 'uniform': All neighbors weighted equally
- 'distance': Closer neighbors weighted more
- Distance weighting reduces influence of far neighbors

### Algorithm
- 'auto', 'brute', 'kd_tree', 'ball_tree'
- Choose based on n_samples and n_features

## 6. Learning Curve Analysis
- Plot train vs validation scores vs training set size
- High bias: Both curves plateau low
- High variance: Gap between curves
- Use to diagnose overfitting/underfitting
