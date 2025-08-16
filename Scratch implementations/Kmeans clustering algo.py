import random

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def init_centroids(self, X):
        m, n = X.shape
        return np.array(X[random.randint(0, m - 1)] for _ in range(self.k))

    def _assign_clusters(self, X, centroids):
        clusters = []
        for x in X:
            distances = [np.linalg.norm(x - centroids) for _ in centroids]
            cluster_idx = np.argmin(distances)
            clusters.append(cluster_idx)
        return clusters

    def _update_centroids(self, X, clusters):
        centroids = []
        for i in range(self.k):
            points = X[clusters == i]
            if len(points) == 0:
                centroids.append(np.zeros(X.shape[1]))
            else:
                centroids.append(np.mean(points, axis=0))
        return np.array(centroids)

    def fit_predict(self, X):
        clusters = None
        centroids = self.init_centroids(X)

        for i in range(self.max_iter):
            clusters = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, clusters)

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return clusters
