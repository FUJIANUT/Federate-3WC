import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# FCM Algorithm Implementation
def initialize_centers(X, n_clusters):
    return X[np.random.choice(X.shape[0], n_clusters, replace=False)]

def update_membership(X, centers):
    distance = pairwise_distances(X, centers, metric='euclidean')
    # Adding a small constant to avoid division by zero in membership calculation
    return 1.0 / (np.fmax(distance, 1e-8) ** 2)

def update_centers(X, membership, m=2):
    weights = membership ** m
    return np.dot(weights.T, X) / np.sum(weights.T, axis=1, keepdims=True)

def fuzzy_c_means(X, n_clusters, m=2, max_iter=100, tol=1e-4):
    centers = initialize_centers(X, n_clusters)
    for _ in range(max_iter):
        old_centers = centers.copy()
        membership = update_membership(X, centers)
        centers = update_centers(X, membership, m)
        if np.linalg.norm(centers - old_centers) < tol:
            break
    return centers, membership

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply FCM
centers, membership = fuzzy_c_means(X, n_clusters=4)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=membership.argmax(axis=1), cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200) # Plot cluster centers
plt.title("Fuzzy C-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
