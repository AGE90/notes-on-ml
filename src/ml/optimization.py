"""Descent methods (book ch04).

All solvers record their path so notebooks can plot the way down to the minimum:
`history` has one row per iterate, columns `[theta_0, ..., theta_n, J(theta)]`.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

Array = np.ndarray
Step = Literal["fixed", "exact", "quadratic"]


def quadratic(
    A: Array, b: Array
) -> tuple[Callable[[Array], float], Callable[[Array], Array]]:
    """Cost and gradient of the paraboloid $J(\\theta) = \\frac12 \\theta^T A \\theta - b^T \\theta$.

    Minimizing it is the same as solving $A\\theta = b$ (with $A = X^TX$, $b = X^Ty$).
    """
    return (lambda t: float(0.5 * t @ A @ t - b @ t)), (lambda t: A @ t - b)


def _quadratic_interpolation_step(
    cost: Callable[[Array], float], theta: Array, g: Array, tol: float
) -> float:
    """Pick $\\alpha$ by fitting a parabola to $\\varphi(\\alpha) = J(\\theta - \\alpha g)$ at 0, α3/2, α3."""
    phi = lambda a: cost(theta - a * g)  # noqa: E731
    J1, a3 = phi(0.0), 1.0
    while phi(a3) >= J1:  # halve until we find a point that decreases J
        a3 /= 2
        if a3 < tol / 2:
            return 0.0
    a2 = a3 / 2
    J2, J3 = phi(a2), phi(a3)
    # Newton forward divided differences -> vertex of the interpolating parabola
    h1 = (J2 - J1) / a2
    h2 = (J3 - J2) / (a3 - a2)
    h3 = (h2 - h1) / a3
    a0 = 0.5 * (a2 - h1 / h3) if h3 != 0 else a3
    return a0 if phi(a0) < J3 else a3


def gradient_descent(
    cost: Callable[[Array], float],
    grad: Callable[[Array], Array],
    theta0: Array,
    step: Step = "fixed",
    alpha: float = 0.01,
    A: Array | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[Array, Array]:
    """Steepest descent $\\theta_{k+1} = \\theta_k - \\alpha_k \\nabla J(\\theta_k)$.

    Parameters
    ----------
    step
        How $\\alpha_k$ is chosen:
        ``"fixed"`` uses `alpha` every iteration (trial and error);
        ``"exact"`` uses the exact line search $\\alpha_k = r_k^T r_k / r_k^T A r_k$,
        only valid for a quadratic cost with Hessian `A`;
        ``"quadratic"`` uses parabolic interpolation of $\\varphi(\\alpha)$ (any cost).
    tol
        Stop when $\\|\\nabla J\\| < tol$ (or, for ``"quadratic"``, when no step improves J).

    Returns
    -------
    theta, history
    """
    if step == "exact" and A is None:
        raise ValueError("step='exact' needs the Hessian A of a quadratic cost")
    theta = np.asarray(theta0, dtype=float)
    history = [np.append(theta, cost(theta))]
    for _ in range(max_iter):
        g = grad(theta)
        if np.linalg.norm(g) < tol:
            break
        if step == "fixed":
            a = alpha
        elif step == "exact":
            a = float(g @ g) / float(g @ A @ g)  # type: ignore[operator]
        else:
            a = _quadratic_interpolation_step(cost, theta, g, tol)
            if a == 0.0:
                break
        theta = theta - a * g
        history.append(np.append(theta, cost(theta)))
        if not np.isfinite(history[-1][-1]):
            break  # diverged: alpha too large
    return theta, np.array(history)


def conjugate_gradient(
    A: Array, b: Array, x0: Array | None = None, tol: float = 1e-10
) -> tuple[Array, Array]:
    """Linear conjugate gradient for symmetric positive definite `A`.

    Converges in at most $n$ steps in exact arithmetic. Returns `(x, path)`,
    where `path` stacks every iterate.
    """
    x = np.zeros_like(b, dtype=float) if x0 is None else np.asarray(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    path = [x.copy()]
    for _ in range(len(b)):
        if np.linalg.norm(r) < tol:
            break
        Ap = A @ p
        a = (r @ r) / (p @ Ap)
        x = x + a * p
        r_new = r - a * Ap
        p = r_new + (r_new @ r_new) / (r @ r) * p
        r = r_new
        path.append(x.copy())
    return x, np.array(path)
