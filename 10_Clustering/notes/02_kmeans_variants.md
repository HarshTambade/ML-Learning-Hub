# K-Means and Variants

## K-Means Algorithm
Iteratively assigns points to nearest centroid and updates centroids until convergence.

## K-Means++
Smarter initialization: chooses centers with probability proportional to distance squared, reducing poor local minima.

## Mini-Batch K-Means
Uses random subsets for faster processing on large datasets with slight quality trade-off.

## K-Medoids (PAM)
Uses actual data points as centers (medoids) instead of means, more robust to outliers.

## Fuzzy C-Means
Soft clustering where each point has membership probability for all clusters (0-1 range).

## Time Complexity
- K-Means: O(n*k*d*iterations)
- K-Means++: O(k*d*n) initialization + K-Means

## Parameter Tuning
- K value: Use elbow method or silhouette score
- Initialization: K-Means++ recommended
- n_init: Number of random initializations
