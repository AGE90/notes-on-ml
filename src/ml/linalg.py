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


def pca(X: Array, r: int) -> tuple[Array, Array, Array]:
    """PCA of `X` (rows are observations) through the SVD of the centered data.

    Parameters
    ----------
    X : Array
        Data matrix of shape (n_observations, n_features).
    r : int
        Number of principal components to keep.

    Returns
    -------
    tuple[Array, Array, Array]
        Scores $Z = X_c V_r$, principal directions $V_r$ (columns) and all the
        singular values $s$ of $X_c$; the covariance eigenvalues are
        $s^2 / (n - 1)$.
    """
    Xc = X - X.mean(axis=0)
    _, s, VT = np.linalg.svd(Xc, full_matrices=False)
    V_r = VT[:r].T
    return Xc @ V_r, V_r, s


def dft_matrix(n: int) -> Array:
    """Unitary DFT matrix $F_{kn} = e^{-j 2 \\pi k n / N} / \\sqrt{N}$.

    Parameters
    ----------
    n : int
        Signal length $N$.

    Returns
    -------
    Array
        Complex matrix of shape (n, n) with $F^H F = I$; $F\\mathbf{x}$ equals
        `np.fft.fft(x, norm="ortho")`.
    """
    k = np.arange(n)
    F: Array = np.exp(-2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)
    return F


def matmul_naive(A: Array, B: Array) -> Array:
    """Matrix product with three pure-Python loops: the $O(n^3)$ baseline BLAS beats.

    Parameters
    ----------
    A : Array
        Matrix of shape (n, p).
    B : Array
        Matrix of shape (p, m).

    Returns
    -------
    Array
        $C = AB$ of shape (n, m).
    """
    a, b = A.tolist(), B.tolist()
    n, p, m = len(a), len(b), len(b[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik, row = a[i][k], b[k]
            for j in range(m):
                C[i][j] += aik * row[j]
    return np.array(C)
