# Chapter 10: Clustering Projects

## Overview
These projects provide real-world applications of clustering algorithms. Each project spans from beginner to advanced difficulty levels and includes domain-specific challenges.

---

## Project 1: Customer Segmentation for E-Commerce

**Difficulty Level:** Beginner to Intermediate

**Domain:** Business Analytics / Marketing

### Objective
Segment e-commerce customers based on purchasing behavior to enable targeted marketing strategies.

### Dataset
Use a public e-commerce dataset (e.g., Online Retail dataset) or create synthetic data with:
- Customer ID
- Purchase frequency
- Average order value
- Last purchase date
- Product categories purchased
- Lifetime value

### Tasks
1. **Data Exploration:**
   - Analyze customer distribution
   - Identify feature correlations
   - Handle missing values

2. **Feature Engineering:**
   - Create RFM features (Recency, Frequency, Monetary)
   - Calculate customer lifetime value
   - Normalize features appropriately

3. **Clustering:**
   - Try K-Means with optimal k
   - Try Hierarchical clustering
   - Compare silhouette scores

4. **Segment Characterization:**
   - Profile each segment (demographics, behavior)
   - Create visualization of segments
   - Develop marketing strategies for each segment

### Expected Outcome
3-5 distinct customer segments with clear business characteristics and actionable insights.

---

## Project 2: Image Compression using K-Means

**Difficulty Level:** Intermediate

**Domain:** Computer Vision / Image Processing

### Objective
Reduce image file size using K-Means color quantization (reducing color palette).

### Dataset
Any image file (JPEG, PNG) - larger images with many colors work best.

### Tasks
1. **Image Loading and Preprocessing:**
   - Load image as numpy array
   - Reshape to 2D array of pixel colors
   - Normalize RGB values to [0, 1]

2. **K-Means Color Quantization:**
   - Apply K-Means with k = 8, 16, 32, 64 colors
   - Track compression ratio and quality trade-offs
   - Time the clustering process

3. **Reconstruction:**
   - Replace each pixel with nearest centroid color
   - Reshape back to image dimensions
   - Save compressed images

4. **Analysis:**
   - Visual comparison of original vs compressed
   - Calculate compression ratio (file size reduction)
   - Evaluate quality metrics (PSNR, MSE)
   - Plot quality vs compression trade-off curve

### Expected Outcome
Visual demonstration of color quantization effectiveness and understanding of image compression trade-offs.

---

## Project 3: Document/Text Clustering and Topic Discovery

**Difficulty Level:** Intermediate to Advanced

**Domain:** Natural Language Processing / Text Mining

### Objective
Cluster text documents to discover latent topics or document groups.

### Dataset
Use a document collection (e.g., 20 Newsgroups, BBC News, scientific papers) or collect news articles.

### Tasks
1. **Text Preprocessing:**
   - Remove special characters, numbers
   - Convert to lowercase
   - Remove stopwords
   - Lemmatization/Stemming

2. **Feature Extraction:**
   - TF-IDF vectorization
   - LSA (Latent Semantic Analysis)
   - Optional: Word embeddings (Word2Vec, GloVe)

3. **Clustering:**
   - Apply K-Means on TF-IDF vectors
   - Try hierarchical clustering
   - Try DBSCAN on embeddings

4. **Topic Analysis:**
   - Extract top terms for each cluster
   - Assign meaningful topic names
   - Visualize clusters (t-SNE or UMAP reduction)
   - Calculate coherence scores

5. **Evaluation:**
   - Use external metrics if labeled data available
   - Measure topic quality

### Expected Outcome
Identified document clusters with interpretable topics and meaningful groupings.

---

## Project 4: Gene Sequence Clustering and Genetic Similarity Detection

**Difficulty Level:** Advanced

**Domain:** Bioinformatics / Computational Biology

### Objective
Cluster DNA/protein sequences to identify genetic similarities and evolutionary relationships.

### Dataset
Public sequence databases (e.g., GenBank, UniProt) or download from NCBI.

### Tasks
1. **Sequence Data Preparation:**
   - Parse FASTA format sequences
   - Select organism group for clustering
   - Handle sequence alignments

2. **Feature Extraction:**
   - K-mer encoding (subsequence frequencies)
   - Position-specific scoring matrices
   - Evolutionary distance metrics (Hamming, edit distance)
   - Optional: Deep learning embeddings

3. **Clustering:**
   - Apply K-Means to k-mer features
   - Try spectral clustering on similarity matrix
   - Use DBSCAN for density-based grouping

4. **Phylogenetic Analysis:**
   - Create dendrogram from hierarchical clustering
   - Compare with known phylogenetic trees
   - Identify novel evolutionary relationships

5. **Validation:**
   - Cross-reference with NCBI taxonomy
   - Calculate biological clustering quality
   - Visualize evolutionary distances

### Expected Outcome
Cluster phylogenetically related sequences with biological validity.

---

## Project 5: Anomaly Detection in Network Traffic

**Difficulty Level:** Advanced

**Domain:** Cybersecurity / Network Analysis

### Objective
Detect network anomalies (attacks, intrusions, unusual patterns) using clustering.

### Dataset
Use KDD Cup 99 dataset, NSL-KDD, CICIDS2017, or synthetically generated network traffic.

### Tasks
1. **Feature Engineering:**
   - Extract network flow features (packet counts, duration, protocols)
   - Compute statistical summaries
   - Normalize high-dimensional features

2. **Baseline Normal Behavior:**
   - Cluster "normal" traffic samples
   - Characterize normal cluster center and boundaries
   - Calculate distance thresholds

3. **Anomaly Detection:**
   - Apply DBSCAN to detect outliers
   - Use isolation forest (ensemble method)
   - Calculate distances to normal cluster
   - Flag anomalies based on thresholds

4. **Evaluation:**
   - Calculate True Positive Rate (TPR)
   - Calculate False Positive Rate (FPR)
   - Plot ROC curve
   - Analyze detected anomalies

5. **Visualization:**
   - Show network traffic in reduced dimension (t-SNE)
   - Highlight normal vs anomalous traffic
   - Create temporal anomaly patterns

### Expected Outcome
Effective anomaly detection system with quantified detection rates and false alarm rates.

---

## Project 6: Social Network Community Detection

**Difficulty Level:** Advanced

**Domain:** Social Network Analysis / Graph Analytics

### Objective
Identify communities (groups of closely connected nodes) in social networks.

### Dataset
Public social networks (Zachary's Karate Club, Dolphin Network, Les Misérables) or collect from APIs.

### Tasks
1. **Network Construction:**
   - Build adjacency matrix from network data
   - Calculate similarity/distance metrics
   - Handle weighted/unweighted edges

2. **Graph Clustering Methods:**
   - Spectral clustering on graph Laplacian
   - Louvain algorithm (modularity optimization)
   - Hierarchical clustering on similarity matrix

3. **Feature Engineering:**
   - Node embeddings using random walks
   - Structural features (clustering coefficient, betweenness)
   - Deep graph embeddings (GraphSAGE, GCN)

4. **Community Detection:**
   - Identify optimal number of communities
   - Assign nodes to communities
   - Calculate modularity score

5. **Validation and Analysis:**
   - Compare with known community structure
   - Calculate Normalized Mutual Information (NMI)
   - Analyze inter/intra-community connections
   - Visualize network with community coloring

6. **Interpretation:**
   - Characterize community properties
   - Identify influential nodes
   - Understand network dynamics

### Expected Outcome
Identified communities matching (or improving upon) known network structure with quantified validation metrics.

---

## General Project Guidelines

### Documentation Requirements
1. **Problem Statement:** Clear description of objective
2. **Data Description:** Dataset source, features, preprocessing
3. **Methodology:** Algorithms used, parameter selection, validation approach
4. **Results:** Metrics, visualizations, interpretations
5. **Conclusions:** Key findings, insights, recommendations

### Code Quality
- Well-commented and organized code
- Reproducible results (fixed random seeds)
- Error handling and edge case management
- Comparison of multiple approaches

### Visualizations
- 2D/3D scatter plots of clusters
- Silhouette plots
- Dendrograms (if hierarchical)
- Domain-specific visualizations
- Quality metrics plots

### Validation Metrics
- Internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- External metrics (if labeled data available)
- Domain-specific evaluation measures
- Multiple validation approaches

## Resources

- Scikit-learn clustering documentation
- Domain-specific libraries:
  - NLP: NLTK, spaCy, Gensim
  - Bioinformatics: BioPython, BLAST
  - Networks: NetworkX, igraph
  - Images: OpenCV, PIL, scikit-image
- Research papers on clustering applications
- Kaggle competitions for inspiration

## Difficulty Progression Path

1. **Start with Project 1** - Customer Segmentation (fundamental concepts)
2. **Move to Project 2** - Image Compression (apply to new domain)
3. **Try Project 3** - Text Clustering (handle high-dimensional data)
4. **Challenge yourself with Project 4 or 5** - Advanced domain applications
5. **Master Project 6** - Graph-based clustering (most complex)

Each project builds on clustering concepts while introducing domain-specific challenges. Complete at least 2-3 projects to gain comprehensive practical experience.
