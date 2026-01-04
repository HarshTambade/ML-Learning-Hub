# Clustering Fundamentals

## Definition
Clustering is an unsupervised learning technique that partitions data into groups (clusters) where:
- Similar items are grouped together
- Dissimilar items are separated
- No predefined labels required

## Key Concepts

### Distance Metrics
- **Euclidean**: sqrt(sum((xi-yi)^2))
- **Manhattan**: sum(|xi-yi|)
- **Cosine**: 1 - (dot product / (||x|| * ||y||))

### Similarity vs Dissimilarity
- Similarity: High when items are alike
- Dissimilarity/Distance: Low when items are alike

## Clustering Quality

### Internal Metrics (No ground truth)
- **Silhouette Score**: -1 to 1 (higher better)
- **Davies-Bouldin Index**: Lower better
- **Calinski-Harabasz**: Higher better

### External Metrics (With labels)
- **Adjusted Rand Index**: -1 to 1
- **Normalized Mutual Information**: 0 to 1

## Advantages
- Discover hidden patterns
- No labeled data required
- Reduces dimensionality
- Data segmentation

## Limitations
- Evaluation without labels is challenging
- Sensitive to outliers
- Algorithm selection matters
- Requires preprocessing (scaling)
