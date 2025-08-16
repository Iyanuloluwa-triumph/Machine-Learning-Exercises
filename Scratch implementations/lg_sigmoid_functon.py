import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
print(dir(digits))

print(digits.data.shape)
X = digits.data
y = digits.target

# Binary classification 0 vs 1
mask = (y == 0) | (y == 1)
X_binary = X[mask]
y_binary = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize weight and bias
weights = np.zeros(X_train.shape[1], dtype=float)
b = 0
losses = []


# define sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Binary cross-entropy loss
def compute_loss(X, y, w, b):
    z = np.dot(X, w) + b
    h = sigmoid(z)
    loss = -np.mean(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
    return loss


# Initialize LogisticRegression model
def train_binary(X_fit, y_fit, w, b, lr=0.01, epochs=1000):
    for i in range(epochs):
        z = np.dot(X_fit, w) + b
        h = sigmoid(z)
        error = h - y_fit

        dw = np.dot(X_fit.T, error) / len(y_fit)
        db = np.sum(error)
        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            print(f"epochs {i}, weight {w[::20]}, bias {b}, loss {compute_loss(X_fit, y_fit, w, b):.4f}")
        loss = compute_loss(X_fit, y_fit, w, b)
        losses.append(loss)
    return w, b


# Train model
w, b = train_binary(X_train, y_train, weights, b)

# loss plot
plt.plot(range(1000), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.grid()
plt.show()


def predict_binary(X, w, b):
    return (sigmoid(np.dot(X, w) + b) >= 0.5).astype(int)


# test accuracy
y_pred = predict_binary(X_test, w, b)
accuracy = np.mean(y_pred == y_test)
print(f"test accuracy: {accuracy * 100:.2f}%")
