"""Dynamical systems and Dynamic Mode Decomposition.

The vector fields follow the `f(t, x)` signature of `scipy.integrate.solve_ivp`.
"""

import numpy as np

Array = np.ndarray


def damped_pendulum(t: float, x: Array, c: float = 0.5, w2: float = 4.0) -> Array:
    r"""Pendulum $\ddot\theta + c\dot\theta + \omega^2 \sin\theta = 0$, with `x` $= (\theta, \dot\theta)$."""
    return np.array([x[1], -c * x[1] - w2 * np.sin(x[0])])


def van_der_pol(t: float, x: Array, mu: float = 1.0) -> Array:
    r"""Van der Pol oscillator $\ddot x - \mu(1 - x^2)\dot x + x = 0$; stable limit cycle for $\mu > 0$."""
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def lorenz(
    t: float, x: Array, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3
) -> Array:
    """Lorenz system; the default parameters give the chaotic strange attractor."""
    return np.array(
        [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    )


def slow_manifold_system(t: float, x: Array, mu: float, lam: float) -> Array:
    r"""System $\dot x_1 = \mu x_1$, $\dot x_2 = \lambda (x_2 - x_1^2)$.

    It has the invariant manifold $x_2 = \lambda / (\lambda - 2\mu) \, x_1^2$ and
    becomes linear in the observables $(x_1, x_2, x_1^2)$.
    """
    return np.array([mu * x[0], lam * (x[1] - x[0] ** 2)])


def dmd(X: Array, Xp: Array, r: int) -> tuple[Array, Array]:
    """Exact Dynamic Mode Decomposition of the snapshot pairs $X' \\approx AX$.

    Parameters
    ----------
    X : Array
        Snapshots $\\mathbf{x}_1, \\ldots, \\mathbf{x}_{m-1}$ as columns.
    Xp : Array
        The same snapshots one time step later, $\\mathbf{x}_2, \\ldots, \\mathbf{x}_m$.
    r : int
        Truncation rank of the SVD of `X`.

    Returns
    -------
    tuple[Array, Array]
        Discrete-time eigenvalues and DMD modes $\\Phi$ (columns).
    """
    U, s, VH = np.linalg.svd(X, full_matrices=False)
    U, s, V = U[:, :r], s[:r], VH[:r].conj().T
    A_tilde = U.conj().T @ Xp @ V / s
    eigenvalues, W = np.linalg.eig(A_tilde)
    return eigenvalues, Xp @ (V / s) @ W
