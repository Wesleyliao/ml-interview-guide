import numpy as np

class Kmeans:
    def __init__(self, k, seed=0):
        self.k = k
        self.seed = seed
        self.centroids = None  # (k, d)

    def calc_pairwise_distances(self, X, centroids):
        # Calculate squared distance from every point to every center.
        # X: (n, d); centroids: (k, d); returns: (n, k).
        # (n, 1, d) - (1, k, d) -> (n, k, d) -> (n, k)
        # Squared distance has the same argmin; no sqrt needed.
        return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    def get_assignments(self, distances):
        # Assign each point to its nearest center; returns (n,).
        return np.argmin(distances, axis=1)

    def get_updated_centroids(self, X, assignments):
        # Make each center the mean of its assigned points; returns (k, d).
        # Keep the previous center if no points were assigned to it.
        centers = self.centroids.copy()
        for i in range(self.k):
            points = X[assignments == i]
            if len(points):  # Keep previous center for an empty cluster.
                centers[i] = points.mean(axis=0)
        return centers

    def fit(self, X, max_iter=100, tol=1e-6):
        # X: (n, d). Convert to float so centroid means are not truncated.
        X = np.asarray(X, dtype=float)
        if not 1 <= self.k <= len(X) or max_iter < 1:
            raise ValueError('Require 1 <= k <= n and max_iter >= 1')
        # 1. Assign initial centroids by sampling k data points.
        rng = np.random.default_rng(self.seed)
        self.centroids = X[rng.choice(len(X), self.k, replace=False)].copy()
        # 2. Alternate assignments and centroid updates until convergence.
        for _ in range(max_iter):
            # Calculate pairwise distances and assign nearest centers.
            labels = self.predict(X)
            # Make each centroid the average of the points assigned to it.
            updated = self.get_updated_centroids(X, labels)
            # Check whether centers stopped moving (or hit max_iter).
            converged = np.max(np.abs(updated - self.centroids)) <= tol
            self.centroids = updated
            if converged:
                break
        # 3. Compute final loss: sum of squared distances to nearest centers.
        self.inertia_ = np.min(self.calc_pairwise_distances(X, self.centroids), axis=1).sum()
        return self

    def predict(self, X):
        # Calculate distances, then return nearest-center labels: (n,).
        return self.get_assignments(self.calc_pairwise_distances(X, self.centroids))
