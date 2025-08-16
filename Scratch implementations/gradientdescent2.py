import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def gradient_descent(x_val, y_val):
    m_curr, b_curr = 0, 0
    iterations = 100
    learning_rate = 0.03
    n = len(x_val)
    cost = 0
    cost_history = []

    for i in range(iterations):
        y_predicted = m_curr * x_val + b_curr
        old_cost = cost
        cost = 1 / n * sum(value ** 2 for value in (y_val - y_predicted))
        cost_history.append(cost)
        grad_m = -2 / n * sum(x_val * (y_val - y_predicted))
        grad_b = -2 / n * sum(y_val - y_predicted)
        m_curr = m_curr - learning_rate * grad_m
        b_curr = b_curr - learning_rate * grad_b
        print("m {}, b {}, iteration {}, cost {}".format(m_curr, b_curr, i, cost))
        if math.isclose(cost, old_cost, rel_tol=1e-20):
            break
    plt.plot(range(iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("cost(MSE)")
    plt.title("cost against Iterations in gradient descent")
    plt.grid(True)
    plt.show()


x = np.array([1, 3, 6, 9])
y = np.array([1, 7, 13, 19])

gradient_descent(x, y)

reg = LinearRegression()
reg.fit(x.reshape(-1, 1), y)

print(reg.coef_, reg.intercept_)
