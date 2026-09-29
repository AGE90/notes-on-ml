"""Draw the machine learning mind map -> reports/figures/ml_mind_map.{png,svg}.

Edit MAP below (and COVERED when a topic gets a notebook), then run:

    uv run python scripts/make_mind_map.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

from ml.utils.paths import reports_figures_dir

# branch -> [(leaf, examples)]; first half of the branches goes left, the rest right
MAP = {
    "Supervised\nlearning": [
        ("Regression", "linear, polynomial, ridge/lasso"),
        ("Classification", "logistic, SVM, k-NN, naive Bayes"),
        ("Tree ensembles", "random forest, gradient boosting"),
    ],
    "Unsupervised\nlearning": [
        ("Clustering", "k-means, HDBSCAN, GMM"),
        ("Dimensionality reduction", "PCA/SVD, t-SNE, UMAP"),
        ("Anomaly detection", "Gaussian density, isolation forest"),
    ],
    "Self-supervised\nlearning": [
        ("Contrastive learning", "SimCLR, CLIP"),
        ("Masked modeling", "BERT, masked autoencoders"),
        ("Next-token prediction", "GPT-style pretraining"),
    ],
    "Reinforcement\nlearning": [
        ("Value-based", "Q-learning, DQN"),
        ("Policy gradients", "REINFORCE, PPO"),
        ("Model-based", "planning, world models"),
        ("Learning from feedback", "RLHF, RLAIF"),
    ],
    "Deep learning\narchitectures": [
        ("MLP", "forward propagation, backprop"),
        ("CNN", "images, signals"),
        ("RNN / LSTM", "sequences"),
        ("Transformers", "attention, LLMs"),
        ("Graph neural networks", "molecules, networks"),
    ],
    "Generative\nmodels": [
        ("VAE", "latent variable models"),
        ("GAN", "generator vs. discriminator"),
        ("Diffusion models", "image, audio generation"),
        ("Foundation models", "LLMs, fine-tuning, in-context"),
    ],
    "Model evaluation\n& selection": [
        ("Metrics", "precision, recall, F1, ROC"),
        ("Validation", "train/CV/test, k-fold"),
        ("Regularization", "L1/L2, bias-variance"),
        ("Feature selection", "filter, wrapper, embedded"),
        ("Hyperparameter tuning", "grid, random, genetic"),
    ],
}
# leaves with a notebook in notebooks/ (see notes/ROADMAP.md)
COVERED = {
    "Regression",
    "Classification",
    "MLP",
    "Metrics",
    "Regularization",
    "Feature selection",
    "Hyperparameter tuning",
}

COLORS = ["#4C72B0", "#55A868", "#8172B3", "#C44E52", "#DD8452", "#937860", "#2A9D8F"]
LEAF_GAP, GROUP_GAP = 1.0, 0.7
X_BRANCH, X_LEAF = 5.5, 11.8
W_ROOT, W_BRANCH, W_LEAF = 2.8, 3.9, 5.3


def curve(ax, p0, p1, color, lw):
    """Horizontal S-shaped Bezier from p0 to p1."""
    xm = (p0[0] + p1[0]) / 2
    path = Path(
        [p0, (xm, p0[1]), (xm, p1[1]), p1],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, alpha=0.7, zorder=1))


def box(ax, xy, w, h, fc, ec, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] - w / 2, xy[1] - h / 2),
            w,
            h,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=2,
        )
    )


def main() -> None:
    branches = list(MAP.items())
    half = (len(branches) + 1) // 2
    sides = [(-1, branches[:half]), (1, branches[half:])]
    height = max(
        len(sum((v for _, v in bs), [])) * LEAF_GAP + (len(bs) - 1) * GROUP_GAP
        for _, bs in sides
    )

    fig, ax = plt.subplots(figsize=(20, height * 0.62))
    color_iter = iter(COLORS)
    for sign, bs in sides:
        n_leaves = sum(len(v) for _, v in bs)
        y = (n_leaves * LEAF_GAP + (len(bs) - 1) * GROUP_GAP) / 2 - LEAF_GAP / 2
        for name, leaves in bs:
            color = next(color_iter)
            ys = [y - i * LEAF_GAP for i in range(len(leaves))]
            y = ys[-1] - LEAF_GAP - GROUP_GAP
            bxy = (sign * X_BRANCH, sum(ys) / len(ys))
            curve(
                ax,
                (sign * W_ROOT / 2, 0),
                (bxy[0] - sign * W_BRANCH / 2, bxy[1]),
                color,
                4,
            )
            box(ax, bxy, W_BRANCH, 0.9, color, color)
            ax.text(
                *bxy,
                name,
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                weight="bold",
                zorder=3,
            )
            for (leaf, examples), ly in zip(leaves, ys, strict=True):
                lxy = (sign * X_LEAF, ly)
                covered = leaf in COVERED
                curve(
                    ax,
                    (bxy[0] + sign * W_BRANCH / 2, bxy[1]),
                    (lxy[0] - sign * W_LEAF / 2, ly),
                    color,
                    2,
                )
                box(
                    ax,
                    lxy,
                    W_LEAF,
                    0.72,
                    color + ("55" if covered else "18"),
                    color,
                    lw=2.5 if covered else 1,
                )
                ax.text(
                    lxy[0],
                    ly + 0.14,
                    ("✓ " if covered else "") + leaf,
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold" if covered else "normal",
                    zorder=3,
                )
                ax.text(
                    lxy[0],
                    ly - 0.19,
                    examples,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#444",
                    zorder=3,
                )

    box(ax, (0, 0), W_ROOT, 1.3, "#264653", "#264653")
    ax.text(
        0,
        0,
        "Machine\nLearning",
        ha="center",
        va="center",
        color="white",
        fontsize=17,
        weight="bold",
        zorder=3,
    )
    ax.text(
        0,
        -height / 2 - 0.3,
        "✓ bold = covered by a notebook in this knowledge base",
        ha="center",
        fontsize=10,
        color="#444",
    )
    ax.set(
        xlim=(-X_LEAF - W_LEAF / 2 - 0.3, X_LEAF + W_LEAF / 2 + 0.3),
        ylim=(-height / 2 - 0.7, height / 2 + 0.3),
        aspect="equal",
    )
    ax.axis("off")
    for ext in ("png", "svg"):
        fig.savefig(
            reports_figures_dir(f"ml_mind_map.{ext}"),
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
        )
    print("wrote", reports_figures_dir("ml_mind_map.png"))


if __name__ == "__main__":
    main()
