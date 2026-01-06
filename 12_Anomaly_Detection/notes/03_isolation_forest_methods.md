# Isolation Forest Method

## Overview

Isolation Forest is a tree-based anomaly detection algorithm that isolates anomalies instead of profiling normal points.

## Key Principle

Anomaly Isolation: "Anomalies are isolated by randomly selecting a feature and a value, then recursing until all points are isolated."

## Algorithm

1. **Random Feature Selection**: Pick random features
2. **Random Split Values**: Create random splits
3. **Build Isolation Trees**: Recursively partition data
4. **Calculate Anomaly Score**: Based on path length
5. **Threshold Decision**: Points with short paths are anomalies

## Why It Works

- **Anomalies need fewer splits**: They're isolated quicker
- **Normal points need more splits**: Spread across space
- **Path length is key metric**: Shorter path = more anomalous

## Advantages

1. **Fast**: O(n log n) - linear time complexity
2. **Scalable**: Handles high-dimensional data
3. **No Distance Metrics**: Doesn't use distance calculations
4. **Low Memory**: Random sampling of features
5. **Works with Collective Anomalies**: Not just point anomalies
6. **No Hyperparameter Tuning**: Robust defaults

## Disadvantages

1. **Less Interpretable**: Black-box nature
2. **Struggles with Dense Data**: All points look similar
3. **Limited Explanability**: Hard to explain why

## Parameters

- **n_estimators**: Number of trees (default: 100)
- **max_samples**: Samples per tree (default: 'auto')
- **contamination**: Expected anomaly proportion
- **max_features**: Features per split
- **random_state**: Reproducibility

## When to Use

- High-dimensional data
- Large datasets
- Mixed point and collective anomalies
- When speed is critical
- Limited domain knowledge

## Implementation Tips

1. **Standardize features**: Before fitting
2. **Set contamination**: Based on domain knowledge
3. **Tune max_samples**: Trade-off speed vs. accuracy
4. **Cross-validate**: Use multiple anomalies datasets
5. **Ensemble**: Combine with other algorithms

---
*Next: Local Outlier Factor*
