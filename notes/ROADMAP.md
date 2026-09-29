# Study Roadmap

Reference: *Machine Learning for Science and Engineering, Vol. I – Fundamentals* (Jaramillo & Rüger). In Claude Code, `/ml-knowledge chXX` loads a chapter summary.

Legend: ✅ notebook done · 🟡 partial · ⬜ to do

## Phase 1 — Foundations and linear models (done)

| Book | Topic | Status | Notebook |
|---|---|---|---|
| ch01 | What is ML, paradigms | ✅ | [00/01](../notebooks/00_foundations/01_what_is_ml.ipynb), [00/02](../notebooks/00_foundations/02_supervised_learning.ipynb) |
| ch02 | Eigenvalues, SVD, condition number | 🟡 | [00/04 SVD](../notebooks/00_foundations/04_svd.ipynb) |
| ch03 | Least squares, pseudo-inverse, Tikhonov | ✅ | [00/03](../notebooks/00_foundations/03_linear_inversion_problem.ipynb), [01/08](../notebooks/01_linear_regression/08_review.ipynb) |
| ch04 | Normal equations, gradient descent, CG | ✅ | [01/01–05](../notebooks/01_linear_regression/) |
| ch05 | Feature scaling, polynomial regression, overfitting | ✅ | [01/06](../notebooks/01_linear_regression/06_multivariate_and_feature_scaling.ipynb), [01/07](../notebooks/01_linear_regression/07_polynomial_regression.ipynb) |
| ch06 | Logistic regression | ✅ | [02/01–04](../notebooks/02_logistic_regression/) |
| ch07 | Classification metrics | ✅ | [03/01](../notebooks/03_classification_metrics/01_confusion_matrix_precision_recall_f1.ipynb) |

## Phase 2 — Neural networks and model selection (in progress)

| Book | Topic | Status | Next step |
|---|---|---|---|
| ch08 | Forward propagation, XOR | ✅ | [04/01](../notebooks/04_neural_networks/01_forward_propagation.ipynb) |
| ch08 | Cost function and **backpropagation** | ⬜ | `04_neural_networks/02_backpropagation.ipynb`: derive $D^{(l)} = (D^{(l+1)}\Theta^{(l)T}) \circ A^{(l)}(1 - A^{(l)})$, add `backward()` to `ml/neural_net.py` with a gradient check, train on XOR and digits |
| ch04 | **SGD and minibatch** | ⬜ | add `sgd()` to `ml/optimization.py`; compare batch vs. SGD vs. minibatch on a large regression |
| ch09 | Train/CV/test, bias–variance, **learning curves** | 🟡 | `05_model_selection/03_learning_curves.ipynb`: $J_{train}$, $J_{cv}$ vs. $m$ and vs. $\lambda$ |
| ch05, ch09 | Feature selection, hyperparameter tuning | ✅ | [05/01](../notebooks/05_model_selection/01_feature_selection.ipynb), [05/02](../notebooks/05_model_selection/02_hyperparameter_tuning.ipynb) |
| ch02 | Eigen-decomposition, spectral radius, Rayleigh quotient | ⬜ | extend `00_foundations/04_svd` or add `05_eigen` |

## Phase 3 — Other supervised models

| Book | Topic | Status | Suggested notebook |
|---|---|---|---|
| ch10 | SVM: max margin, hinge loss, kernels | ⬜ | `06_svm/01_max_margin.ipynb`, `02_kernel_trick.ipynb` |
| ch11 | Decision trees (Gini), random forests, choosing a classifier | ⬜ | `07_trees/01_cart.ipynb`, `02_random_forest.ipynb` |
| ch11 | CNN, GAN | ⬜ | `08_deep_learning/` |

## Phase 4 — Unsupervised learning and systems

| Book | Topic | Status | Suggested notebook |
|---|---|---|---|
| ch12 | $k$-means, silhouette, HDBSCAN | ⬜ | `09_clustering/` |
| ch13 | PCA ($Z = XW$, 99 % energy rule) | ⬜ | `10_pca/`, reuse `ml.linalg.energy` |
| ch14 | Anomaly detection (Gaussian, $\varepsilon$ by $F_1$) | ⬜ | `11_anomaly_detection/`, reuse `ml.metrics.f1` |
| ch15 | Recommender systems, photo OCR pipeline | ⬜ | `12_recommenders/` |
| ch16 | Counting (multichoose), environments | ⬜ | reference only |

## Session log

Add a line per study session (newest first), linking the note in `sessions/`.

| Date | Topic | Note |
|---|---|---|
| 2026-09-28 | Reorganized the knowledge base | – |
