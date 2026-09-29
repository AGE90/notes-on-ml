"""Activation functions and logistic regression (book ch06).

`X` is expected to already contain the bias column (see `ml.linear_models.add_bias`).
"""

import numpy as np

from ml.optimization import gradient_descent

Array = np.ndarray
EPS = 1e-12  # keeps log() finite when h saturates to 0 or 1


def sigmoid(z: Array) -> Array:
    """$S(z) = 1 / (1 + e^{-z})$, with $S'(z) = S(z)(1 - S(z))$."""
    return 0.5 * (
        1.0 + np.tanh(0.5 * np.asarray(z, dtype=float))
    )  # same as 1/(1+e^-z), no overflow


def tanh(z: Array) -> Array:
    """$\\tanh z = 2 / (1 + e^{-2z}) - 1 = 2S(2z) - 1$."""
    return np.tanh(z)


def relu(z: Array) -> Array:
    """$\\mathrm{ReLU}(z) = \\max\\{0, z\\}$."""
    out: Array = np.maximum(0.0, np.asarray(z, dtype=float))
    return out


def predict_proba(theta: Array, X: Array) -> Array:
    """$h_\\theta(x) = S(X\\theta) = p(y = 1 \\mid x, \\theta)$."""
    return sigmoid(X @ theta)


def predict(theta: Array, X: Array, threshold: float = 0.5) -> Array:
    """Binary prediction: 1 if $h_\\theta(x) \\ge \\gamma$ (threshold), else 0."""
    return (predict_proba(theta, X) >= threshold).astype(int)


def cross_entropy_cost(theta: Array, X: Array, y: Array, lam: float = 0.0) -> float:
    """$J(\\theta) = -\\frac1m \\sum y\\log h + (1-y)\\log(1-h) + \\frac{\\lambda}{2m}\\|\\theta_{1:}\\|^2$."""
    m = len(y)
    h = np.clip(predict_proba(theta, X), EPS, 1 - EPS)
    J = -(y @ np.log(h) + (1 - y) @ np.log(1 - h)) / m
    return float(J + lam / (2 * m) * theta[1:] @ theta[1:])


def cross_entropy_gradient(theta: Array, X: Array, y: Array, lam: float = 0.0) -> Array:
    """$\\nabla J = \\frac1m X^T (h - y)$ (+ $\\frac{\\lambda}{m}\\theta$, bias not regularized)."""
    m = len(y)
    g: Array = X.T @ (predict_proba(theta, X) - y) / m
    g[1:] += lam / m * theta[1:]
    return g


def fit_logistic(
    X: Array,
    y: Array,
    alpha: float = 0.1,
    lam: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-6,
) -> tuple[Array, Array]:
    """Fit logistic regression by batch gradient descent. Returns `(theta, history)`."""
    return gradient_descent(
        lambda t: cross_entropy_cost(t, X, y, lam),
        lambda t: cross_entropy_gradient(t, X, y, lam),
        np.zeros(X.shape[1]),
        step="fixed",
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )


def one_vs_all_fit(X: Array, y: Array, **kwargs: float) -> Array:
    """Train one binary classifier per class ("class k" vs "the rest").

    Returns a matrix with one column of parameters per class (sorted by label).
    """
    return np.column_stack(
        [fit_logistic(X, (y == k).astype(float), **kwargs)[0] for k in np.unique(y)]  # type: ignore[arg-type]
    )


def one_vs_all_predict(Theta: Array, X: Array) -> Array:
    """Pick the class whose classifier gives the highest $h_\\theta^{(k)}(x)$ (labels 0..K-1)."""
    return np.argmax(sigmoid(X @ Theta), axis=1)
