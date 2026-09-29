"""Feature normalization and polynomial features (book ch05).

Each scaler returns the scaled data *and* a function that applies the same
transform to new data: the parameters fitted on the training set must be reused
at prediction time. Never scale the bias column (add it afterwards).
"""

from collections.abc import Callable

import numpy as np

Array = np.ndarray
Transform = Callable[[Array], Array]


def _fit(X: Array, shift: Array, scale: Array) -> tuple[Array, Transform]:
    scale = np.where(scale == 0, 1.0, scale)  # constant column: leave it unscaled
    transform: Transform = lambda Z: (np.asarray(Z, dtype=float) - shift) / scale  # noqa: E731
    return transform(X), transform


def standardize(X: Array) -> tuple[Array, Transform]:
    """z-score: $x_j := (x_j - \\mu_j) / \\sigma_j$."""
    X = np.asarray(X, dtype=float)
    return _fit(X, X.mean(axis=0), X.std(axis=0))


def mean_normalize(X: Array) -> tuple[Array, Transform]:
    """$x_j := (x_j - \\mu_j) / (\\max x_j - \\min x_j)$."""
    X = np.asarray(X, dtype=float)
    return _fit(X, X.mean(axis=0), np.ptp(X, axis=0))


def min_max(X: Array) -> tuple[Array, Transform]:
    """$x_j := (x_j - \\min x_j) / (\\max x_j - \\min x_j)$, values in $[0, 1]$."""
    X = np.asarray(X, dtype=float)
    return _fit(X, X.min(axis=0), np.ptp(X, axis=0))


def unit_norm(X: Array) -> tuple[Array, Transform]:
    """$x_j := x_j / \\|x_j\\|$ (each column gets norm 1)."""
    X = np.asarray(X, dtype=float)
    return _fit(X, np.zeros(X.shape[1]), np.linalg.norm(X, axis=0))


def polynomial_features(x: Array, degree: int) -> Array:
    """Columns $[x, x^2, \\dots, x^d]$ of a single feature (no bias column)."""
    x = np.asarray(x, dtype=float).ravel()
    return np.column_stack([x**k for k in range(1, degree + 1)])


def polynomial_features_2d(X: Array, degree: int) -> Array:
    """All monomials $x_1^a x_2^b$ with $1 \\le a + b \\le d$ of two features (no bias)."""
    x1, x2 = np.asarray(X, dtype=float).T
    return np.column_stack(
        [x1 ** (k - j) * x2**j for k in range(1, degree + 1) for j in range(k + 1)]
    )
