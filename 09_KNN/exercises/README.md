# KNN Exercises

## Exercise 1: Basic KNN Classification
**Objective**: Understand how KNN works with different K values

### Tasks:
1. Load the Iris dataset
2. Split into train-test sets (80-20)
3. Train KNN models with K=3, 5, 7
4. Evaluate using accuracy, precision, recall, F1-score
5. Plot decision boundaries
6. Compare performance across different K values

### Expected Learning Outcomes:
- Understand the effect of K on model bias-variance
- Learn how to split and evaluate datasets
- Know how to visualize KNN decision boundaries

---

## Exercise 2: Distance Metrics Comparison
**Objective**: Compare different distance metrics

### Tasks:
1. Use KNN with Euclidean, Manhattan, and Chebyshev distances
2. Evaluate on same dataset with same K value
3. Measure execution time for each metric
4. Plot performance metrics comparison
5. Analyze which metric works best

### Expected Learning Outcomes:
- Understand when to use different distance metrics
- Learn performance implications of metric choice
- Practice benchmarking different approaches

---

## Exercise 3: Feature Scaling Impact
**Objective**: Understand the importance of feature scaling

### Tasks:
1. Train KNN on unscaled features
2. Train KNN on StandardScaler normalized features
3. Train KNN on MinMaxScaler normalized features
4. Compare accuracies
5. Visualize feature distributions

### Expected Learning Outcomes:
- Understand why scaling matters for KNN
- Learn different scaling techniques
- Know when to apply normalization

---

## Exercise 4: Cross-Validation and Hyperparameter Tuning
**Objective**: Find optimal K using cross-validation

### Tasks:
1. Perform 5-fold cross-validation for K=1 to 20
2. Plot CV scores vs K values
3. Identify optimal K
4. Compare CV score with test set performance
5. Use GridSearchCV to automate the process

### Expected Learning Outcomes:
- Understand cross-validation importance
- Learn to tune hyperparameters systematically
- Know the difference between CV and test scores

---

## Exercise 5: KNN for Regression
**Objective**: Apply KNN to regression problems

### Tasks:
1. Load a regression dataset (Boston Housing or similar)
2. Train KNN regressor with different K values
3. Evaluate using MSE, RMSE, MAE, R² score
4. Plot actual vs predicted values
5. Compare with other regression algorithms

### Expected Learning Outcomes:
- Apply KNN to regression tasks
- Understand regression-specific evaluation metrics
- Compare algorithm performance for regression

---

## Exercise 6: Handling Imbalanced Data
**Objective**: Deal with imbalanced classification datasets

### Tasks:
1. Create an imbalanced dataset
2. Train KNN without handling imbalance
3. Use class_weight='balanced' in KNN
4. Apply SMOTE oversampling
5. Compare F1 scores and confusion matrices

### Expected Learning Outcomes:
- Identify imbalanced data problems
- Learn multiple approaches to handle imbalance
- Understand appropriate metrics for imbalanced data
