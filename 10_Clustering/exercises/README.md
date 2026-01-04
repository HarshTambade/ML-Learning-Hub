# Chapter 10: Clustering - Comprehensive Exercises

## Overview
These exercises cover fundamental and advanced clustering concepts, from K-Means basics to density-based and probabilistic clustering methods. Each exercise builds practical skills and theoretical understanding.

---

## Exercise 1: K-Means Clustering Basics

**Objective:** Understand K-Means algorithm and implement it from scratch and using scikit-learn.

**Learning Outcomes:**
- Implement K-Means step-by-step
- Understand centroids and cluster assignments
- Visualize clustering results

**Tasks:**
1. Generate synthetic data (e.g., 3 Gaussian clusters with 100 points each)
2. Implement K-Means manually:
   - Random initialization of centroids
   - Assignment step (assign points to nearest centroid)
   - Update step (recalculate centroids)
   - Iterate until convergence
3. Compare with scikit-learn's KMeans
4. Plot clusters and centroids on 2D scatter plot
5. Track inertia changes across iterations

**Expected Outcome:** Understanding of K-Means mechanics and convergence behavior.

---

## Exercise 2: Elbow Method for Optimal k

**Objective:** Use elbow method and silhouette analysis to determine optimal number of clusters.

**Learning Outcomes:**
- Calculate inertia and silhouette scores
- Interpret elbow curves
- Make data-driven decisions on cluster count

**Tasks:**
1. Load a real-world dataset (e.g., iris, wine, or breast cancer)
2. For k = 1 to 10:
   - Fit K-Means with k clusters
   - Calculate inertia
   - Calculate silhouette score
3. Plot inertia vs k (look for elbow point)
4. Plot silhouette score vs k (look for peak)
5. Identify optimal k using both methods
6. Create visualization comparing both metrics

**Expected Outcome:** Clear understanding of when to stop adding clusters.

---

## Exercise 3: Distance Metrics Comparison

**Objective:** Understand how different distance metrics affect clustering results.

**Learning Outcomes:**
- Compare Euclidean, Manhattan, and Cosine distances
- Understand metric sensitivity to data properties
- Choose appropriate metrics for different data types

**Tasks:**
1. Create or load a dataset
2. Apply K-Means with different distance metrics:
   - Euclidean (L2)
   - Manhattan (L1)
   - Cosine similarity
3. For each metric:
   - Compute cluster assignments
   - Calculate cluster quality metrics
   - Visualize results if possible
4. Compare results:
   - Are cluster memberships the same?
   - How do metrics differ for normalized vs raw data?

**Expected Outcome:** Appreciation for metric choice impact on results.

---

## Exercise 4: Impact of Feature Scaling

**Objective:** Understand the critical importance of feature normalization in clustering.

**Learning Outcomes:**
- Recognize scale-dependent nature of distance metrics
- Apply standardization and normalization correctly
- Evaluate impact on clustering quality

**Tasks:**
1. Load a multi-scale dataset (features with very different ranges)
2. Cluster without preprocessing
3. Cluster after StandardScaler normalization
4. Cluster after MinMaxScaler normalization
5. Compare silhouette scores and visualizations
6. Document how different features influence clustering

**Expected Outcome:** Understanding that preprocessing is essential, not optional.

---

## Exercise 5: Hierarchical Clustering

**Objective:** Implement and interpret hierarchical clustering with dendrograms.

**Learning Outcomes:**
- Understand agglomerative clustering mechanism
- Interpret dendrograms
- Compare different linkage methods

**Tasks:**
1. Create small synthetic dataset (20-30 points for clarity)
2. Apply agglomerative clustering with different linkage methods:
   - Single linkage
   - Complete linkage
   - Average linkage
   - Ward linkage
3. For each linkage, create dendrogram
4. Cut each dendrogram at different heights
5. Compare cluster assignments from different linkages
6. Identify which linkage best separates your clusters

**Expected Outcome:** Ability to read dendrograms and choose appropriate linkage.

---

## Exercise 6: DBSCAN Parameter Exploration

**Objective:** Understand DBSCAN parameters and their impact on clustering.

**Learning Outcomes:**
- Master eps and min_samples parameters
- Use k-distance graph for eps selection
- Identify noise points (outliers)

**Tasks:**
1. Generate data with varying densities (e.g., moons or circles dataset)
2. Use k-distance graph method:
   - Compute k-distance for all points
   - Plot sorted k-distances
   - Identify elbow for eps value
3. Test multiple eps values:
   - Too small (too many noise points)
   - Too large (clusters merge)
   - Just right (good separation)
4. Vary min_samples parameter
5. Visualize:
   - Core points
   - Border points
   - Noise points (-1 label)
6. Calculate clustering quality metrics

**Expected Outcome:** Practical skill in DBSCAN parameter selection.

---

## Exercise 7: Gaussian Mixture Models (Soft Clustering)

**Objective:** Understand probabilistic clustering and soft assignments.

**Learning Outcomes:**
- Understand EM algorithm concept
- Interpret membership probabilities
- Use BIC/AIC for model selection

**Tasks:**
1. Fit Gaussian Mixture Model to data
2. Get hard assignments (predict)
3. Get soft assignments (predict_proba):
   - Show probability matrix
   - Identify ambiguous assignments (close to 0.5)
4. Compare hard vs soft assignments
5. For k = 1 to 10:
   - Fit GMM
   - Calculate BIC and AIC
6. Plot BIC/AIC vs k
7. Visualize 2D data with cluster probability contours

**Expected Outcome:** Understanding of probabilistic vs hard clustering.

---

## Exercise 8: Comprehensive Clustering Comparison

**Objective:** Compare all clustering methods on same dataset.

**Learning Outcomes:**
- Understand strengths/weaknesses of each algorithm
- Choose appropriate algorithm for problem
- Validate results using multiple metrics

**Tasks:**
1. Load or create dataset with interesting structure
2. Apply clustering algorithms:
   - K-Means (with optimal k from Exercise 2)
   - Hierarchical (best linkage from Exercise 5)
   - DBSCAN (parameters from Exercise 6)
   - GMM (optimal k from Exercise 7)
3. For each algorithm, calculate:
   - Silhouette score
   - Davies-Bouldin index
   - Calinski-Harabasz index
   - Runtime
4. Create comparison table
5. Visualize results (2D projection if needed)
6. Discuss:
   - Which algorithm works best for this data?
   - Why do you think it performs better?
   - What are the trade-offs?

**Expected Outcome:** Practical ability to select and apply appropriate clustering algorithm.

---

## General Tips for All Exercises

1. **Always standardize data** - Use StandardScaler or normalize features
2. **Visualize results** - Plotting is crucial for understanding
3. **Validate thoroughly** - Use multiple quality metrics
4. **Document your process** - Write clear comments and explanations
5. **Handle edge cases** - Consider empty clusters, convergence issues
6. **Compare implementations** - Verify your code against scikit-learn
7. **Test on different data** - Understand algorithm behavior across datasets

## Evaluation Metrics Summary

| Metric | What it measures | Good value |
|--------|------------------|------------|
| Silhouette Score | How well-separated clusters are | Closer to 1.0 |
| Davies-Bouldin | Cluster dispersion/separation | Closer to 0 |
| Calinski-Harabasz | Cluster density/separation ratio | Higher is better |
| Inertia | Sum of squared distances | Lower is better |
| Homogeneity | All samples same class in clusters | Closer to 1.0 |
| Completeness | All same class samples in cluster | Closer to 1.0 |

## Resources

- Scikit-learn clustering documentation
- Your lecture notes on clustering theory
- Code examples in code_examples folder
- Theory in notes folder for deep understanding
