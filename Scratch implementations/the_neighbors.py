from collections import Counter

import numpy as np


def euclidean_distances(x, y):
    return np.linalg.norm(x - y)


class KNNs:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # just store the values (lazy learner)
        self.X = X
        self.y = y

    def predict(self, X):
        # return an array of predictions based on nearest neighbors
        neighbors = [self._predict(x) for x in X]
        return np.array(neighbors)

    def _predict(self, point):
        # calculate euclidean distances
        distances = [euclidean_distances(point, x) for x in self.X]

        # store smallest distance indexes and their labels
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y[i] for i in k_indices]

        # which label has the majority?
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
