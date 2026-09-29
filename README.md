# Notes on ML

Personal machine learning knowledge base: theory, algorithms and experiments

Theory (with $\LaTeX$ math) and visualizations in notebooks, reproducible algorithms in `src/ml/`, study-session notes and experiments. The reference curriculum is *Machine Learning for Science and Engineering, Vol. I* (Jaramillo & Rüger), available to Claude Code as the `/ml-knowledge` skill.

**Start here → [`notes/ROADMAP.md`](notes/ROADMAP.md)** (study plan and progress).

## Knowledge map

| # | Topic | Notebooks | Algorithms (`src/ml/`) |
|---|---|---|---|
| 00 | Foundations | [what is ML](notebooks/00_foundations/01_what_is_ml.ipynb) · [supervised learning](notebooks/00_foundations/02_supervised_learning.ipynb) · [linear inversion](notebooks/00_foundations/03_linear_inversion_problem.ipynb) · [SVD](notebooks/00_foundations/04_svd.ipynb) | `linalg.py` |
| 01 | Linear regression | [normal equations](notebooks/01_linear_regression/01_normal_equations.ipynb) · [GD theory](notebooks/01_linear_regression/02_gradient_descent_theory.ipynb) · [fixed step](notebooks/01_linear_regression/03_gd_fixed_step.ipynb) · [exact line search](notebooks/01_linear_regression/04_gd_exact_line_search.ipynb) · [quadratic interp., CG, BFGS](notebooks/01_linear_regression/05_gd_quadratic_interp_cg_bfgs.ipynb) · [multivariate & scaling](notebooks/01_linear_regression/06_multivariate_and_feature_scaling.ipynb) · [polynomial & overfitting](notebooks/01_linear_regression/07_polynomial_regression.ipynb) · [review](notebooks/01_linear_regression/08_review.ipynb) | `linear_models.py`, `optimization.py`, `features/scaling.py` |
| 02 | Logistic regression | [hypothesis & activations](notebooks/02_logistic_regression/01_hypothesis_and_activations.ipynb) · [cost & gradient](notebooks/02_logistic_regression/02_cost_and_gradient.ipynb) · [decision boundary](notebooks/02_logistic_regression/03_decision_boundary.ipynb) · [one-vs-all](notebooks/02_logistic_regression/04_one_vs_all.ipynb) | `logistic.py` |
| 03 | Classification metrics | [confusion matrix, P/R/F1, ROC](notebooks/03_classification_metrics/01_confusion_matrix_precision_recall_f1.ipynb) | `metrics.py` |
| 04 | Neural networks | [forward propagation](notebooks/04_neural_networks/01_forward_propagation.ipynb) | `neural_net.py` |
| 05 | Model selection | [feature selection](notebooks/05_model_selection/01_feature_selection.ipynb) · [hyperparameter tuning](notebooks/05_model_selection/02_hyperparameter_tuning.ipynb) | (scikit-learn) |

Shared plots (cost surfaces with descent paths, convergence curves, decision boundaries) live in `src/ml/visualization/plots.py`.

## Study workflow

1. Pick the next topic in the [roadmap](notes/ROADMAP.md); read the book chapter (`/ml-knowledge chXX` in Claude Code).
2. Write the session note: copy [`notes/sessions/_template.md`](notes/sessions/_template.md) to `notes/sessions/YYYY-MM-DD-<topic>.md`.
3. Theory + visual intuition go in a topic notebook (`notebooks/NN_topic/`); any algorithm you implement goes in `src/ml/` with a test in `tests/unit/test_algorithms.py`, and the notebook imports it.
4. Free-form trials go in `notebooks/experiments/YYYY-MM-DD-<topic>.ipynb`; log runs with MLflow when comparing settings.
5. Tick the roadmap.

The original course notes (Spanish) and the older `ml-topics` project were merged into these notebooks; they remain in the git history (first commit) if needed.


---

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.12 for you if it is missing.

```bash
git clone <repository-url>
cd notes-on-ml
make install                  # uv sync --all-groups: creates .venv with every dependency group
uv run pre-commit install     # optional: lint and format on every commit
```

Run any command inside the environment with `uv run <command>`, or activate it with `source .venv/bin/activate`. Add dependencies with `uv add <package>` (or `uv add --group <dev|test|notebook|data-science|viz> <package>`) and update them with `uv lock --upgrade && uv sync`.

---

## Usage

Run `make help` to list every task.

The template's data → features → train pipeline (`make data/features/train/predict`) is kept for experiments on real datasets; see `make help`.

### Code quality and tests

```bash
make check    # ruff format + ruff check + mypy
make test     # pytest with coverage
```

### Paths and data

Never hardcode paths. The helpers in `utils/paths.py` resolve from the project root, so they work the same in scripts, notebooks and tests:

```python
from ml.data.data_loader import load_csv
from ml.utils.paths import data_raw_dir, reports_figures_dir
from ml.visualization.visualize import plot_distribution

df = load_csv(data_raw_dir("dataset.csv"))
plot_distribution(
    df["size"],
    title="Size distribution",
    xlabel="size",
    save_path=reports_figures_dir("size.png"),
)
```

### Notebooks

Start Jupyter Lab with `make notebook`. Put reusable code in `src/` and import it; add this at the top of a notebook to pick up code changes without restarting the kernel:

```python
%load_ext autoreload
%autoreload 2
```

### Experiment tracking (MLflow)

`make train` logs parameters, metrics and the model to MLflow with `log_mlflow_experiment` (in `models/model_utils.py`). Runs are stored in `mlflow.db` and `mlruns/` at the project root (both git-ignored). Browse them with:

```bash
make mlflow-ui    # http://127.0.0.1:5000
```

---

## Project Structure

```text
├── CLAUDE.md               <- Conventions for Claude Code
├── notes/
│   ├── ROADMAP.md          <- Study plan: book chapters -> notebooks, progress
│   └── sessions/           <- One markdown note per study session
├── notebooks/
│   ├── 00_foundations/ … 05_model_selection/   <- Topic notebooks (theory + code + plots)
│   └── experiments/        <- Free experiments, dated
├── src/ml/
│   ├── linalg.py           <- SVD helpers, condition number
│   ├── linear_models.py    <- Bias column, normal equations, ridge, MSE cost/gradient
│   ├── optimization.py     <- Gradient descent (fixed / exact / quadratic step), conjugate gradient
│   ├── features/scaling.py <- Standardize, min-max, mean/unit-norm scaling, polynomial features
│   ├── logistic.py         <- Sigmoid/tanh/ReLU, cross-entropy, logistic regression, one-vs-all
│   ├── metrics.py          <- Confusion matrix, precision, recall, F1, ROC/AUC
│   ├── neural_net.py       <- Forward propagation
│   ├── visualization/      <- `plots.py` (study plots) and template EDA helpers
│   └── data/, models/, utils/paths.py  <- Template pipeline and path helpers
├── data/raw/               <- ECG signals and images used by the notebooks (git-ignored)
├── reports/figures/        <- Figures used in the notes (mind map from `scripts/make_mind_map.py`, network diagrams, …)
├── references/             <- Papers, manuals and other reference material
└── tests/unit/             <- `test_algorithms.py` checks every algorithm
```

---

## Documentation

- [Developer Guide](docs/developer_guide.md): code style, testing, Git workflow and contributing
- [Code of Conduct](docs/code_of_conduct.md)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
