"""Binary classification metrics (book ch07). Positive class = 1."""

import numpy as np

Array = np.ndarray


def confusion_matrix(y_true: Array, y_pred: Array) -> Array:
    """`[[TN, FP], [FN, TP]]` (rows = truth, columns = prediction, like sklearn)."""
    t, p = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return np.array(
        [[np.sum(~t & ~p), np.sum(~t & p)], [np.sum(t & ~p), np.sum(t & p)]]
    )


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def accuracy(y_true: Array, y_pred: Array) -> float:
    """$(TP + TN) / m$. Misleading on imbalanced data."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true: Array, y_pred: Array) -> float:
    """$P = TP / (TP + FP)$: of what I flagged, how much was right."""
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fp)


def recall(y_true: Array, y_pred: Array) -> float:
    """$R = TP / (TP + FN)$: of the real positives, how many I caught."""
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fn)


def f1(y_true: Array, y_pred: Array) -> float:
    """Harmonic mean $F_1 = 2PR / (P + R)$."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)


def roc_curve(y_true: Array, scores: Array) -> tuple[Array, Array, Array]:
    """False/true positive rates for every threshold. Returns `(fpr, tpr, thresholds)`."""
    y_true = np.asarray(y_true).astype(bool)
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    pred = np.asarray(scores)[None, :] >= thresholds[:, None]
    tpr = (pred & y_true).sum(axis=1) / max(y_true.sum(), 1)
    fpr = (pred & ~y_true).sum(axis=1) / max((~y_true).sum(), 1)
    return fpr, tpr, thresholds


def auc(fpr: Array, tpr: Array) -> float:
    """Area under the ROC curve (trapezoidal rule)."""
    return float(np.trapezoid(tpr, fpr))
