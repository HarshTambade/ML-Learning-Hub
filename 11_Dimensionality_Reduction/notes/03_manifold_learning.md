# Manifold Learning: t-SNE, UMAP, and Nonlinear Techniques

## Manifold Learning Concepts

**Manifold Learning** assumes data lies on a lower-dimensional manifold embedded in high-dimensional space.

### Key Assumptions
1. Data lies on a lower-dimensional surface (manifold)
2. Local structure matters more than global
3. Nonlinear relationships exist in data
4. Preservation of local neighborhoods is important

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Algorithm Overview
1. **Convert distances to probabilities** for each point
2. **Minimize KL divergence** between original and embedded spaces
3. **Student-t distribution** creates repulsive forces
4. **Iterative optimization** to find best layout

### Advantages
- Excellent for visualization
- Reveals clusters and structure
- No hyperparameters affecting compression

### Disadvantages
- Computationally expensive O(n²)
- Distorts global structure
- Non-parametric (can't transform new data)
- Sensitive to hyperparameters

### Key Parameters
- **Perplexity**: Controls local neighborhood (5-50)
- **Learning rate**: Step size for optimization
- **n_iter**: Number of iterations

### When to Use t-SNE
✓ Visualization (2D/3D)
✓ Exploratory data analysis
✓ Understanding cluster structure
✓ Medium-sized datasets

✗ Downstream machine learning
✗ Large datasets (>100k points)
✗ Streaming data
✗ Preserving global distances

## UMAP (Uniform Manifold Approximation and Projection)

### Algorithm Overview
1. **Topological approach** using graph theory
2. **Simplicial complexes** to represent data
3. **Cross-entropy** minimization for embedding
4. **Fast optimization** with stochastic gradient descent

### Advantages
- Preserves more global structure than t-SNE
- Much faster than t-SNE
- Parametric (can transform new data)
- Flexible distance metrics
- Scales to large datasets

### Disadvantages
- More complex than t-SNE
- Requires careful hyperparameter tuning
- Less intuitive than t-SNE

### Key Parameters
- **n_neighbors**: Size of local neighborhood (5-200)
- **min_dist**: Minimum distance between points
- **metric**: Distance metric to use

### When to Use UMAP
✓ Large datasets
✓ Need global structure preservation
✓ Want to transform new data
✓ Visualization + downstream tasks
✓ Streaming/incremental learning

✗ Very small datasets
✗ Highly sensitive to local structure only

## Other Manifold Learning Methods

### Isomap
- Preserves geodesic distances
- Uses k-NN graph
- Applies MDS to embedded distances
- Good for well-separated clusters

### Locally Linear Embedding (LLE)
- Preserves local linear structure
- Reconstructs points from k-NN
- Applies eigenvalue decomposition
- Works well for smooth manifolds

### Spectral Embedding
- Uses eigenvectors of Laplacian
- Assumes data on manifold
- Good for clustering

## Manifold Learning vs PCA

| Aspect | PCA | Manifold Learning |
|--------|-----|-------------------|
| Linear/Nonlinear | Linear | Nonlinear |
| Global/Local | Global | Local |
| Speed | Fast | Slow to medium |
| Scalability | Good | Limited |
| Interpretability | High | Low |
| Visualization | Fair | Excellent |

## Best Practices

1. **Start with UMAP** for most tasks
2. **Use t-SNE** only for visualization
3. **Scale features** before applying
4. **Experiment with parameters** via visualization
5. **Validate** with downstream tasks
6. **Consider data size** - UMAP for large, t-SNE for small

## Common Pitfalls

1. **Over-interpreting structure** - manifold learning creates artifacts
2. **Using for downstream ML** - structure may not be meaningful
3. **Fixed hyperparameters** - different data needs different settings
4. **Ignoring computational cost** - t-SNE slow for large data
5. **Not validating** - visual structure doesn't guarantee model improvement
