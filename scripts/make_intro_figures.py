"""Draw the 00_foundations/01 intro figures in the mind map style.

-> reports/figures/ai_ml_dl.png, reports/figures/classical_vs_ml.png

    uv run python scripts/make_intro_figures.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

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


if __name__ == "__main__":
    ai_ml_dl()
    classical_vs_ml()
