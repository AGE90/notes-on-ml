"""Linear-algebra helpers (book ch02-ch03)."""

import numpy as np

Array = np.ndarray


def low_rank_approx(X: Array, r: int) -> Array:
    """Best rank-`r` approximation $X_r = U_r \\Sigma_r V_r^T$ (Eckart-Young)."""
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    return (U[:, :r] * s[:r]) @ VT[:r]


def energy(s: Array) -> Array:
    """Cumulative energy $\\sum_{i \\le k} \\sigma_i^2 / \\sum_i \\sigma_i^2$ of singular values `s`."""
    s2 = np.asarray(s, dtype=float) ** 2
    e: Array = np.cumsum(s2) / s2.sum()
    return e


def condition_number(X: Array) -> float:
    """$\\kappa = \\sigma_{max} / \\sigma_{min}$: $\\kappa = 10^k$ loses about $k$ digits."""
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0] / s[-1])
