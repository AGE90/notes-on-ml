"""Linear regression: least squares, normal equations and ridge (book ch03-ch04)."""

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (the bias / intercept term) to `X`."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.hstack([np.ones((X.shape[0], 1)), X])


def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve $X^T X \\theta = X^T y$ in the least-squares sense.

    Uses `lstsq` (SVD) instead of an explicit inverse, so it also returns the
    minimum-norm solution $\\theta = X^+ y$ when $X^T X$ is singular.
    """
    theta: np.ndarray = np.linalg.lstsq(X, y, rcond=None)[0]
    return theta


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov / ridge solution $\\theta = (X^T X + \\lambda I)^{-1} X^T y$."""
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(n), X.T @ y)


def mse_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """$J(\\theta) = \\frac{1}{2m}\\|X\\theta - y\\|^2$."""
    r = X @ theta - y
    return float(r @ r) / (2 * len(y))


def mse_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """$\\nabla J(\\theta) = \\frac{1}{m} X^T (X\\theta - y)$."""
    g: np.ndarray = X.T @ (X @ theta - y) / len(y)
    return g
