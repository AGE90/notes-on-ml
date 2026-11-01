"""Draw the intro figures in the mind map style.

-> reports/figures/ai_ml_dl.png, reports/figures/classical_vs_ml.png (00_foundations/01)
-> reports/figures/GD.png (01_linear_regression/02)

    uv run python scripts/make_intro_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch

from ml.optimization import gradient_descent
from ml.utils.paths import reports_figures_dir

ROOT, ML, DL = "#264653", "#4C72B0", "#C44E52"


def save(fig, name: str) -> None:
    fig.savefig(reports_figures_dir(name), bbox_inches="tight", dpi=150, facecolor="white")
    print("wrote", reports_figures_dir(name))


def ai_ml_dl() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    # (center, width, height, color, label, label y, subtitle)
    sets = [
        ((0, 0), 10, 6.6, ROOT, "Artificial intelligence", 2.6, "systems that act intelligently"),
        ((0.9, -0.55), 7.0, 4.6, ML, "Machine learning", 1.15, "learn rules from data"),
        ((1.6, -1.05), 4.5, 2.4, DL, "Deep learning", -0.85, "many-layer neural networks"),
    ]
    for (cx, cy), w, h, color, label, ly, sub in sets:
        ax.add_patch(Ellipse((cx, cy), w, h, fc=color + "26", ec=color, lw=2.5))
        ax.text(cx, ly, label, ha="center", va="center", fontsize=14, weight="bold", color=color)
        ax.text(cx, ly - 0.42, sub, ha="center", va="center", fontsize=9.5, color="#444")
    ax.set(xlim=(-5.2, 5.2), ylim=(-3.5, 3.5), aspect="equal")
    ax.axis("off")
    save(fig, "ai_ml_dl.png")


def classical_vs_ml() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = [(1.6, "Classical\nprogramming", ROOT, ("Rules", "Data"), "Answers"),
            (-1.6, "Machine\nlearning", ML, ("Data", "Answers"), "Rules")]
    arrow = dict(arrowstyle="-|>", lw=2, mutation_scale=18)
    for y, name, color, inputs, output in rows:
        ax.add_patch(FancyBboxPatch((-1.5, y - 0.75), 3, 1.5, boxstyle="round,pad=0.1,rounding_size=0.3",
                                    fc=color, ec=color))
        ax.text(0, y, name, ha="center", va="center", color="white", fontsize=13, weight="bold")
        for dy, inp in zip((0.4, -0.4), inputs, strict=True):
            ax.annotate("", (-1.65, y + dy), (-3.6, y + dy), arrowprops=arrow | {"color": color})
            ax.text(-3.75, y + dy, inp, ha="right", va="center", fontsize=12)
        ax.annotate("", (3.6, y), (1.65, y), arrowprops=arrow | {"color": color})
        ax.text(3.75, y, output, ha="left", va="center", fontsize=12, weight="bold", color=color)
    ax.set(xlim=(-5.6, 5.6), ylim=(-2.7, 2.7), aspect="equal")
    ax.axis("off")
    save(fig, "classical_vs_ml.png")


# Non-convex J: sum of Gaussian hills (a > 0) and pits (a < 0), as (a, center, width)
BUMPS = [(2.2, (0.15, -0.6), 0.55), (1.6, (-1.0, 0.9), 0.6), (-2.0, (-1.1, -1.0), 0.5),
         (-2.4, (1.2, 0.8), 0.6), (-0.8, (1.3, -1.3), 0.45)]


def landscape(t: np.ndarray) -> float:
    return float(sum(a * np.exp(-np.sum((t - c) ** 2) / (2 * w**2)) for a, c, w in BUMPS))


def landscape_grad(t: np.ndarray) -> np.ndarray:
    return sum(-a * (t - np.array(c)) / w**2 * np.exp(-np.sum((t - c) ** 2) / (2 * w**2)) for a, c, w in BUMPS)


def gd_landscape() -> None:
    T0, T1 = np.meshgrid(np.linspace(-2.2, 2.2, 160), np.linspace(-2.2, 2.2, 160))
    J = np.vectorize(lambda a, b: landscape(np.array([a, b])))(T0, T1)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.plot_surface(T0, T1, J, cmap="viridis", rstride=2, cstride=2, lw=0, alpha=0.9, antialiased=True, zorder=1)
    ax.contour(T0, T1, J, levels=15, zdir="z", offset=J.min() - 1, cmap="viridis", linewidths=1, zorder=0)
    # two nearby starts on the hill roll into different pits
    for start, color in [((0.0, -0.75), ML), ((0.3, -0.45), DL)]:
        _, hist = gradient_descent(landscape, landscape_grad, np.array(start), alpha=0.06, max_iter=60)
        ax.plot(hist[:, 0], hist[:, 1], hist[:, 2] + 0.05, "-o", color=color, ms=3.5, lw=2,
                mec="white", mew=0.8, zorder=5)
        ax.scatter(*hist[0, :2], hist[0, 2] + 0.05, s=90, marker="*", color="gold", ec="k", zorder=6)
        ax.scatter(*hist[-1, :2], hist[-1, 2] + 0.05, s=90, marker="X", color=color, ec="white", zorder=6)
    ax.set_xlabel(r"$\theta_0$", fontsize=13)
    ax.set_ylabel(r"$\theta_1$", fontsize=13)
    ax.set_zlabel(r"$J(\theta_0, \theta_1)$", fontsize=13)
    ax.set_zlim(J.min() - 1, J.max())
    ax.view_init(elev=32, azim=-60)
    ax.set_title("Two nearby starts, two different local minima", fontsize=13, weight="bold", color=ROOT)
    save(fig, "GD.png")


if __name__ == "__main__":
    ai_ml_dl()
    classical_vs_ml()
    gd_landscape()
