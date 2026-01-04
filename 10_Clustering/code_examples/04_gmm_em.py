"""Gaussian Mixture Models with EM Algorithm

Demonstrates GMM, soft clustering, BIC/AIC model selection
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=0.7)
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Fit GMMs with different components
for k in [1, 3, 5]:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    
    idx = k // 2
    axes[idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    axes[idx].set_title(f'GMM with {k} components')

# Model selection
bic_scores = []
for n in range(1, 8):
    gmm = GaussianMixture(n_components=n, random_state=42)
    bic_scores.append(gmm.fit(X_scaled).bic(X_scaled))

plt.figure()
plt.plot(range(1, 8), bic_scores, 'b-o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('Model Selection using BIC')
plt.show()

optimal_k = np.argmin(bic_scores) + 1
print(f'Optimal number of components: {optimal_k}')
