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

# Split data into "nodes" to simulate federated environment
X_splits = np.array_split(X, 3)  # Split data into 3 nodes

# Apply Federated FCM
federated_centers = federated_fuzzy_c_means(X_splits, n_clusters=4)
