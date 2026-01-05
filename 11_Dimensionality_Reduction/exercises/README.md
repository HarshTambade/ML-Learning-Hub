# Dimensionality Reduction Exercises

## Exercise 1: PCA on Iris Dataset

**Objective**: Learn PCA fundamentals and variance explanation

**Tasks**:
1. Load Iris dataset
2. Standardize features
3. Apply PCA with different k values (1, 2, 3, 4)
4. Plot variance explained for each component
5. Visualize data in 2D using first two principal components
6. Calculate cumulative variance explained

**Difficulty**: Beginner
**Time**: 30 minutes

---

## Exercise 2: Feature Selection Comparison

**Objective**: Compare different feature selection methods

**Tasks**:
1. Load breast cancer dataset
2. Apply SelectKBest, RFE, and tree-based selection
3. Compare selected features across methods
4. Train classifier with selected features
5. Compare model performance
6. Analyze why methods selected different features

**Difficulty**: Intermediate
**Time**: 45 minutes

---

## Exercise 3: t-SNE vs UMAP Visualization

**Objective**: Understand visualization differences

**Tasks**:
1. Load high-dimensional dataset (digits, MNIST)
2. Apply t-SNE with different perplexity values
3. Apply UMAP with different n_neighbors
4. Compare visualizations
5. Analyze cluster structure
6. Discuss computational time differences

**Difficulty**: Intermediate
**Time**: 60 minutes

---

## Exercise 4: Autoencoder Dimensionality Reduction

**Objective**: Build and train autoencoders

**Tasks**:
1. Create dataset with known intrinsic dimension
2. Build simple autoencoder
3. Train with different bottleneck sizes
4. Plot reconstruction error vs bottleneck size
5. Analyze latent representations
6. Compare with PCA

**Difficulty**: Advanced
**Time**: 90 minutes
**Requirements**: TensorFlow, Keras

---

## Exercise 5: Pipeline Integration

**Objective**: Integrate dimensionality reduction into ML pipeline

**Tasks**:
1. Load dataset (high-dimensional)
2. Create sklearn Pipeline with:
   - StandardScaler
   - PCA/UMAP
   - Classifier
3. Use cross-validation to find best k
4. Compare to baseline (no reduction)
5. Analyze trade-off: performance vs efficiency

**Difficulty**: Intermediate
**Time**: 45 minutes

---

## Exercise 6: Curse of Dimensionality

**Objective**: Observe and understand curse of dimensionality

**Tasks**:
1. Generate synthetic data with varying dimensions
2. Train classifier on each dimension
3. Plot accuracy vs dimension
4. Analyze distance distribution
5. Apply dimensionality reduction
6. Compare performance before/after

**Difficulty**: Intermediate
**Time**: 60 minutes

---

## Exercise 7: Manifold Learning Analysis

**Objective**: Understand when manifold learning helps

**Tasks**:
1. Create Swiss roll dataset
2. Apply PCA, t-SNE, UMAP, Isomap
3. Compare results visually
4. Measure local structure preservation
5. Discuss which method works best

**Difficulty**: Intermediate
**Time**: 75 minutes

---

## Exercise 8: Anomaly Detection with DR

**Objective**: Use dimensionality reduction for anomaly detection

**Tasks**:
1. Load dataset with anomalies
2. Apply PCA, Isolation Forest
3. Use reconstruction error as anomaly score
4. Compare detection performance
5. Analyze trade-offs

**Difficulty**: Advanced
**Time**: 75 minutes

---

## Solutions

Solutions for these exercises are available in the `solutions/` directory.

To check your answer:
```bash
python solutions/exercise_1.py
```
