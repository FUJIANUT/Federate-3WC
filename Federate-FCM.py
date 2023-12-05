# Re-define the Fuzzy C-Means algorithm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def initialize_centers(X, n_clusters):
    return X[np.random.choice(X.shape[0], n_clusters, replace=False)]


def update_membership(X, centers):
    distance = pairwise_distances(X, centers, metric='euclidean')
    # Adding a small constant to avoid division by zero in membership calculation
    return 1.0 / (np.fmax(distance, 1e-8) ** 2)

def update_centers(X, membership, m=2):
    weights = membership ** m
    return np.dot(weights.T, X) / np.sum(weights.T, axis=1, keepdims=True)

# Federated FCM Algorithm Implementation
def federated_fuzzy_c_means(X_splits, n_clusters, m=2, max_iter=100, global_tol=1e-4, local_tol=1e-3):
    global_centers = initialize_centers(np.vstack(X_splits), n_clusters)
    for _ in range(max_iter):
        old_global_centers = global_centers.copy()
        local_centers = []
        for X in X_splits:
            centers, _ = fuzzy_c_means(X, n_clusters, m=m, max_iter=max_iter, tol=local_tol)
            local_centers.append(centers)
        global_centers = np.mean(local_centers, axis=0)
        if np.linalg.norm(global_centers - old_global_centers) < global_tol:
            break
    return global_centers

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

# Split data into "nodes" to simulate federated environment
X_splits = np.array_split(X, 3)  # Split data into 3 nodes

# Apply Traditional FCM
centers_traditional, _ = fuzzy_c_means(X, n_clusters=4)

# Apply Federated FCM
centers_federated = federated_fuzzy_c_means(X_splits, n_clusters=4)

# Visualization
plt.figure(figsize=(12, 6))

# Traditional FCM Results
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], marker='o', color='gray', alpha=0.5)
plt.scatter(centers_traditional[:, 0], centers_traditional[:, 1], c='blue', marker='x', s=200)
plt.title("Traditional FCM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Federated FCM Results
plt.subplot(1, 2, 2)
for X_split in X_splits:
    plt.scatter(X_split[:, 0], X_split[:, 1], marker='o', alpha=0.5)
plt.scatter(centers_federated[:, 0], centers_federated[:, 1], c='green', marker='x', s=200)
plt.title("Federated FCM Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
