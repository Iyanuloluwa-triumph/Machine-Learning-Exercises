import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # equivalent to 1/C
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ensure labels are -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        # Init weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    # Bias doesn't change here
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate simple linearly separable data
X, y = make_blobs(n_samples=100, centers=2, random_state=123)
y = np.where(y == 0, -1, 1)  # SVM needs labels -1 or 1

svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)
predictions = svm.predict(X)


# Plot
def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--', label='decision boundary')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'r', label='-1 margin')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'g', label='+1 margin')

    plt.legend()
    plt.show()


visualize_svm()
