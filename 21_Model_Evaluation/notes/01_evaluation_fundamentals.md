# Model Evaluation Fundamentals

## What is Model Evaluation?

Model evaluation is the process of assessing how well a machine learning model performs on unseen data. It answers critical questions:
- Does the model generalize well?
- Is the model better than baselines?
- What are the model's strengths and weaknesses?
- Is the model ready for production?

## Why Evaluation Matters

### 1. Preventing Overfitting
- A model can memorize training data without learning general patterns
- Evaluation on separate data reveals if generalization is poor
- Essential for building reliable systems

### 2. Comparing Models
- Provides objective way to compare different approaches
- Helps select best model for deployment
- Quantifies performance trade-offs

### 3. Business Justification
- Demonstrates model adds value
- Quantifies ROI and business impact
- Supports stakeholder decisions

## Key Evaluation Concepts

### Train, Validation, Test Split

**Training Set**: Data used to train the model
- Typically 60-70% of data
- Used for parameter learning
- Model sees this data

**Validation Set**: Data for hyperparameter tuning
- Typically 15-20% of data
- Used for model selection
- Model doesn't learn from but sees this data

**Test Set**: Data for final evaluation
- Typically 15-20% of data
- Completely held out
- Never used in any way during training

### Why Separate Datasets?

1. **Prevents Overfitting**: Training metrics can be misleading
2. **True Generalization**: Test performance reflects real-world usage
3. **Fair Comparison**: Different models evaluated equally
4. **Prevents Information Leakage**: Validation data not used in training

## Classification Metrics

### Confusion Matrix
Foundation for classification metrics:
- **TP (True Positives)**: Correctly predicted positive
- **TN (True Negatives)**: Correctly predicted negative
- **FP (False Positives)**: Incorrectly predicted positive
- **FN (False Negatives)**: Incorrectly predicted negative

### Key Metrics

**Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- Overall correctness
- Misleading with imbalanced data
- Best for balanced classes

**Precision**: TP / (TP + FP)
- Of predicted positives, how many are correct?
- Minimize false alarms
- Important when FP is costly

**Recall**: TP / (TP + FN)
- Of actual positives, how many did we find?
- Minimize missed cases
- Important when FN is costly

**F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Single metric for balanced evaluation
- Good for imbalanced data

## Regression Metrics

**Mean Absolute Error (MAE)**
- Average absolute difference from true value
- Easy to interpret (in same units as target)
- Robust to outliers

**Mean Squared Error (MSE)**
- Average squared difference
- Emphasizes large errors
- Sensitive to outliers

**Root Mean Squared Error (RMSE)**
- Square root of MSE
- In same units as target
- More interpretable than MSE

**R² Score**
- Proportion of variance explained
- Range: 0 to 1 (or negative)
- 1 is perfect, 0 is no better than mean

## Data Leakage

Information from test set inadvertently used in training:

### Common Forms
1. **Preprocessing Before Split**: Scaling on full dataset
2. **Feature Engineering on Full Data**: Creating features using all data
3. **Time Series**: Training on future data
4. **Cross-validation Mistakes**: Using test data in preprocessing

### Prevention
- Always split first
- Fit preprocessors on training data only
- Use pipelines to prevent leakage
- Be careful with feature engineering

## Baseline and Sanity Checks

### Why Baselines Matter
- Provides comparison point
- Can't judge if model is good without reference
- Catches obvious issues

### Common Baselines
- **Classification**: Predict most common class
- **Regression**: Predict mean value
- **Time Series**: Predict previous value (naive forecast)

### Sanity Checks
1. Model should beat baseline
2. Features should make sense
3. Performance should be stable
4. Metrics should be reasonable
5. Error analysis reveals patterns

## Evaluation in Different Scenarios

### Imbalanced Classification
- Use F1, precision-recall, or AUC-ROC
- Not accuracy (misleading)
- Consider business costs

### Time Series
- Use walk-forward validation
- Not random k-fold
- Respect temporal order

### Ranking/Recommendation
- Use NDCG, MAP, recall@k
- Position matters
- Partial credit possible

## Summary

Proper evaluation is foundation of ML practice. Always:
- Separate train/validation/test data
- Use appropriate metrics
- Check for data leakage
- Compare against baselines
- Validate on unseen data
