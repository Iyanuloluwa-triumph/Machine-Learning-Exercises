from collections import Counter

import numpy as np


class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _gini(self, y):
        impurity = 1
        classes = np.unique(y)
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            impurity -= p ** 2
        return impurity

    def _information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        gain = self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))
        return gain

    def _best_split(self, X, y, depth=0):
        best_gain = -1
        split_idx, thresh = None, None
        m, n = X.shape
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_split = X[:, feature_index] < threshold
                right_split = X[:, feature_index] >= threshold

                if len(left_split) == 0 or len(right_split) == 0:
                    continue

                y_left, y_right = y[left_split], y[right_split]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    thresh = threshold
        return split_idx, thresh

    def _build_tree(self, X, y, depth=0):
        num_labels = np.unique(y)

        if len(num_labels) == 1 or len(X) == 0 or depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None:
            return Node(value=self._most_common_label(y))

        right_split = X[:, feature_index] >= threshold
        left_split = X[:, feature_index] < threshold

        right = self._build_tree(X[right_split], y[right_split], depth + 1)
        left = self._build_tree(X[left_split], y[left_split], depth + 1)

        return Node(feature=feature_index, threshold=threshold, right=right, left=left)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def score(self, X, y):
        accuracy = np.mean(self.predict(X) == y)
        return accuracy


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


tree = DecisionTree()
np.random.seed(110)
X = np.random.randint(50, 101, (50, 3))
y = np.random.randint(0, 11, (50,))
tree.fit(X, y)
print(tree.predict([[63, 54, 78], [72, 65, 57], [80, 89, 78]]))
print(tree.score([[63, 54, 78], [72, 65, 57], [80, 89, 78]], [6]))
