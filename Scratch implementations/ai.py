import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate simple linear data
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(20) * 2  # y = 2x + 1 + noise

# Fit model without outliers
model_clean = LinearRegression().fit(X, y)
y_pred_clean = model_clean.predict(X)

# Add an outlier (extreme point)
X_outlier = np.append(X, [[10]], axis=0)
y_outlier = np.append(y, [80])  # way above the trend

# Fit model with outlier
model_outlier = LinearRegression().fit(X_outlier, y_outlier)
y_pred_outlier = model_outlier.predict(X_outlier)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Original Data")
plt.scatter([10], [80], color="red", label="Outlier", s=100, marker="x")

# Plot regression lines
plt.plot(X, y_pred_clean, label="Regression (no outlier)", color="green", linewidth=2)
plt.plot(X_outlier, y_pred_outlier, label="Regression (with outlier)", color="orange", linewidth=2)

plt.legend()
plt.title("Effect of Outliers on Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
