import numpy as np


def sigmoid(z):
    # Overflow-safe sigmoid: 1 / (1 + exp(-z)) == exp(-logaddexp(0, -z)).
    return np.exp(-np.logaddexp(0, -z))


def loss_and_gradient(X, y, w, b):
    # X: (n, d), y: (n,) in {0, 1}, w: (d,), b: scalar.
    # 1. Linear scores (logits) and predicted probabilities.
    z = X @ w + b
    p = sigmoid(z)
    # 2. Stable binary cross-entropy and gradients: dL/dz = (p - y) / n.
    loss = np.logaddexp(0, (1 - 2 * y) * z).mean()
    error = p - y
    return loss, (X.T @ error) / len(y), error.mean()


class LogisticRegressor:
    def __init__(self, learning_rate=0.1, steps=1000):
        self.learning_rate = learning_rate
        self.steps = steps

    def fit(self, X, y):
        # 1. Initialize weights and bias to zero.
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.losses = []
        # 2. Run full-batch gradient descent.
        for _ in range(self.steps):
            loss, dw, db = loss_and_gradient(X, y, self.w, self.b)
            self.losses.append(loss)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
