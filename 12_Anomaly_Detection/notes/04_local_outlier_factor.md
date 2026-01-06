# Local Outlier Factor (LOF)

## Overview

LOF is a density-based anomaly detection algorithm. It compares local density of a point to local densities of neighbors.

## Key Concept

**Local Density**: How tightly clustered data is around a point. Anomalies have significantly lower local density than neighbors.

## Algorithm Steps

1. **Calculate K-distance**: Distance to k-th nearest neighbor
2. **Reachability Distance**: Accounts for density variations
3. **Local Reachability Density (LRD)**: Inverse of avg. reachability distance
4. **LOF Score**: Ratio of LRD to neighbors' LRD
5. **Anomaly Decision**: LOF >> 1 indicates anomaly

## Key Characteristics

- **LOF = 1**: Point has similar density to neighbors (normal)
- **LOF > 1**: Point has lower density than neighbors (anomaly)
- **LOF << 1**: Point has higher density (core point)

## Advantages

1. **Detects Local Anomalies**: Points normal globally but anomalous locally
2. **Varying Density**: Works when density changes across space
3. **Multivariate**: Works with multiple features
4. **No Assumptions**: Doesn't assume data distribution
5. **Interpretable**: Local context matters

## Disadvantages

1. **Slow**: O(n²) - quadratic complexity
2. **k-Sensitivity**: Highly dependent on k value
3. **High Dimensions**: Performance degrades
4. **Dense Data**: All points look like neighbors
5. **Hyperparameter Tuning**: Requires good k selection

## Parameters

- **n_neighbors**: Number of neighbors (typically 5-50)
- **metric**: Distance metric (euclidean, manhattan, etc.)
- **contamination**: Expected anomaly proportion
- **novelty**: True for out-of-sample detection

## When to Use

- Varying local density
- When context matters
- Small-to-medium datasets
- Multivariate data
- Need interpretability

## Parameter Tuning Tips

1. **k selection**: Balance between too small and too large
2. **Distance metric**: Choose based on data type
3. **Cross-validation**: Test multiple k values
4. **Ensemble**: Combine with other algorithms

---
*Next: Deep Learning Approaches*
