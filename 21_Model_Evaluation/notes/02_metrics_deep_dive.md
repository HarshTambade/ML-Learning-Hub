# Metrics Deep Dive

## Understanding Evaluation Metrics

Metrics quantify model performance and guide decision-making. Different metrics suit different problems and business contexts.

## Classification Metrics in Detail

### Binary Classification Metrics

#### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Interpretation**: Fraction of correct predictions
- **Use When**: Classes are balanced, all errors equally costly
- **Avoid When**: Classes are imbalanced
- **Example**: Email spam detection (similar spam/non-spam ratio)

#### Precision (Positive Predictive Value)
- **Formula**: TP / (TP + FP)
- **Interpretation**: Of all predicted positive, how many are actually positive?
- **Use When**: False positives are costly
- **Examples**: 
  - Spam filter: High precision means fewer legitimate emails marked spam
  - Medical diagnosis: High precision means fewer false alarms

#### Recall (True Positive Rate, Sensitivity)
- **Formula**: TP / (TP + FN)
- **Interpretation**: Of all actual positives, how many did we find?
- **Use When**: False negatives are costly
- **Examples**:
  - Fraud detection: Recall ensures we catch most fraudulent cases
  - Disease screening: Recall ensures we identify most patients with disease

#### F1-Score
- **Formula**: 2 * (Precision * Recall) / (Precision + Recall)
- **Interpretation**: Harmonic mean of precision and recall
- **Use When**: Need to balance precision and recall
- **Good For**: Imbalanced datasets
- **Range**: 0 to 1 (higher is better)

#### F-Beta Score
- **Formula**: (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
- **When β > 1**: Emphasizes recall
- **When β < 1**: Emphasizes precision
- **Example**: F2-Score (recall twice as important)

#### Specificity (True Negative Rate)
- **Formula**: TN / (TN + FP)
- **Interpretation**: Of all actual negatives, how many did we correctly identify?
- **Complement of**: False Positive Rate
- **Use When**: True negatives matter equally to true positives

#### ROC-AUC (Area Under Receiver Operating Characteristic Curve)
- **Interpretation**: Probability model ranks random positive higher than random negative
- **Range**: 0 to 1 (0.5 = random, 1.0 = perfect)
- **Use When**: Need threshold-independent metric
- **Good For**: Imbalanced data, comparing classifiers
- **Advantage**: Plots TPR vs FPR across all thresholds

#### PR-AUC (Precision-Recall Area Under Curve)
- **Use When**: Class imbalance is severe
- **Advantage**: Better for imbalanced data than ROC-AUC
- **Disadvantage**: Less intuitive than ROC-AUC
- **Compare With**: ROC-AUC when precision matters more

#### Log Loss (Cross-Entropy Loss)
- **Formula**: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
- **Use When**: Probability estimates matter
- **Measures**: Calibration of predicted probabilities
- **Good For**: Ranking models by confidence

### Multi-Class Classification Metrics

#### Macro-Averaging
- Compute metric for each class independently
- Take unweighted mean
- Treats all classes equally
- Suitable when all classes equally important

#### Weighted-Averaging
- Compute metric for each class
- Weight by support (number of samples)
- Handles class imbalance naturally
- Suitable when class frequency varies

#### Micro-Averaging
- Aggregate TP, FP, FN, TN across classes
- Compute metric on aggregated counts
- Equivalent to accuracy for multi-class
- Good overall metric

## Regression Metrics in Detail

### Mean Absolute Error (MAE)
- **Formula**: mean(|y_true - y_pred|)
- **Units**: Same as target variable
- **Interpretation**: Average magnitude of errors
- **Robust to**: Outliers
- **Range**: 0 to infinity

### Mean Squared Error (MSE)
- **Formula**: mean((y_true - y_pred)²)
- **Units**: Target variable squared
- **Interpretation**: Average squared error
- **Sensitive to**: Large errors and outliers
- **Hard to interpret**: Due to squared units

### Root Mean Squared Error (RMSE)
- **Formula**: sqrt(MSE)
- **Units**: Same as target variable
- **Interpretation**: Average error magnitude
- **Compared to**: MAE (RMSE penalizes large errors more)
- **Use When**: Large errors are particularly bad

### R² Score (Coefficient of Determination)
- **Formula**: 1 - (SS_res / SS_tot)
  - SS_res = sum((y_true - y_pred)²)
  - SS_tot = sum((y_true - mean(y_true))²)
- **Range**: Negative to 1 (1 is perfect)
- **Interpretation**: Proportion of variance explained
- **R² = 0**: Model no better than mean
- **R² < 0**: Model worse than mean

### Median Absolute Error
- **Formula**: median(|y_true - y_pred|)
- **Robust to**: Extreme outliers
- **Good For**: Data with significant outliers
- **Drawback**: Less sensitive to distribution shape

## Choosing Metrics: Decision Framework

### For Classification
1. **Is data balanced?**
   - Yes: Use accuracy
   - No: Use F1, ROC-AUC, or PR-AUC

2. **What's more important, FP or FN?**
   - FP: Optimize precision
   - FN: Optimize recall
   - Both: Use F1-Score

3. **Do probability estimates matter?**
   - Yes: Use log loss, Brier score
   - No: Use accuracy, F1, AUC

### For Regression
1. **Are outliers present?**
   - Yes: Use MAE or median AE
   - No: Use RMSE

2. **Need interpretability?**
   - Yes: Use MAE (in same units)
   - No: Use RMSE

3. **Want normalized metric?**
   - Yes: Use R² or MAPE
   - No: Use MAE or RMSE

## Business Metrics vs ML Metrics

### ML Metrics
- Precision, Recall, F1
- RMSE, MAE
- AUC-ROC

### Business Metrics
- Revenue impact
- Cost savings
- User satisfaction
- Time to value

## Summary

Metric selection should be driven by:
1. Problem characteristics (class balance, outliers)
2. Cost of different errors
3. Interpretability requirements
4. Stakeholder priorities

Always use multiple metrics to understand model behavior comprehensively.
