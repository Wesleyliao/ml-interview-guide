import numpy as np

# Your fit/predict structure, with a stable least-squares solve.
class LinearRegressor:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        # X: (n, d), y: (n,). Add intercept, then solve least squares.
        n, d = X.shape
        X = np.concatenate([np.ones((n, 1)), X], axis=1)
        # n >= d+1 does NOT guarantee independent columns.
        self.W = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        # Add the same intercept column, then multiply by fitted weights.
        # Output: (n,).
        n, _ = X.shape
        X = np.concatenate([np.ones((n, 1)), X], axis=1)
        return X @ self.W
