"""Plots reused across the study notebooks."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

Array = np.ndarray


def plot_cost_surface(
    cost: Callable[[Array], float],
    history: Array | None = None,
    t0: tuple[float, float] = (-1, 1),
    t1: tuple[float, float] = (-3, 1),
    zlim: tuple[float, float] | None = None,
    n: int = 100,
    title: str = "",
) -> Figure:
    """3D surface + contour map of $J(\\theta_0, \\theta_1)$, with the descent path in red.

    `history` rows are `[theta_0, theta_1, J]` (as returned by `ml.optimization`).
    """
    T0, T1 = np.meshgrid(np.linspace(*t0, n), np.linspace(*t1, n))
    J = np.array(
        [cost(np.array([a, b])) for a, b in zip(T0.ravel(), T1.ravel(), strict=True)]
    ).reshape(T0.shape)
    zmin = zlim[0] if zlim else J.min() - 0.3 * np.ptp(J)

    fig = plt.figure(figsize=(12, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(T0, T1, J, cmap="viridis", alpha=0.6)
    ax3.contour(T0, T1, J, zdir="z", offset=zmin, cmap="viridis", levels=20)
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(T0, T1, J, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax2, label=r"$J(\theta)$")
    if history is not None:
        ax3.plot(history[:, 0], history[:, 1], history[:, -1], "r.-", zorder=10)
        ax3.plot(history[:, 0], history[:, 1], zmin, "r-")
        ax2.plot(
            history[:, 0],
            history[:, 1],
            "r.-",
            label=f"path ({len(history) - 1} steps)",
        )
        ax2.plot(*history[-1, :2], "r*", ms=14)
        ax2.legend()
    ax3.set_zlim(zmin, zlim[1] if zlim else J.max())
    for ax in (ax3, ax2):
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
    ax3.set_zlabel(r"$J(\theta)$")
    ax2.set_aspect("equal", adjustable="box")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_convergence(
    histories: dict[str, Array], ax: Axes | None = None, log: bool = True
) -> Axes:
    """$J(\\theta_k)$ versus iteration $k$ for one or more runs (the "debug GD" plot)."""
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    for label, h in histories.items():
        ax.plot(h[:, -1], ".-", label=f"{label} ({len(h) - 1} it.)")
    if log:
        ax.set_yscale("symlog")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$J(\theta_k)$")
    ax.legend()
    return ax


def plot_decision_boundary(
    predict: Callable[[Array], Array],
    X: Array,
    y: Array,
    ax: Axes | None = None,
    n: int = 300,
    pad: float = 0.5,
) -> Axes:
    """Shade the regions a 2-feature classifier assigns to each class and scatter the data.

    `predict` maps an `(m, 2)` array of raw features to labels (or probabilities).
    """
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    x1, x2 = X[:, 0], X[:, 1]
    G1, G2 = np.meshgrid(
        np.linspace(x1.min() - pad, x1.max() + pad, n),
        np.linspace(x2.min() - pad, x2.max() + pad, n),
    )
    Z = np.asarray(predict(np.c_[G1.ravel(), G2.ravel()])).reshape(G1.shape)
    k = len(np.unique(y))
    cmap = "coolwarm" if k == 2 else ListedColormap(plt.get_cmap("tab10")(range(k)))
    levels = np.arange(k + 1) - 0.5 if Z.dtype.kind in "iub" else 20
    ax.contourf(G1, G2, Z, alpha=0.25, cmap=cmap, levels=levels)
    ax.scatter(
        x1,
        x2,
        c=y,
        cmap=cmap,
        edgecolors="k",
        s=30,
        vmin=-0.5 if k > 2 else None,
        vmax=k - 0.5 if k > 2 else None,
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    return ax
