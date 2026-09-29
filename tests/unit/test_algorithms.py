import numpy as np
from sklearn import metrics as skm

from ml import metrics
from ml.features.scaling import standardize
from ml.linalg import energy, low_rank_approx
from ml.linear_models import add_bias, normal_equations, ridge
from ml.logistic import (
    cross_entropy_cost,
    cross_entropy_gradient,
    fit_logistic,
    one_vs_all_fit,
    one_vs_all_predict,
)
from ml.neural_net import forward
from ml.optimization import conjugate_gradient, gradient_descent, quadratic

rng = np.random.default_rng(0)
A = np.array([[10.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, -1.0])
THETA_STAR = np.array([0.1, -1.0])


def test_normal_equations_and_ridge():
    X = add_bias(rng.normal(size=(50, 2)))
    y = X @ np.array([1.0, 2.0, -3.0])
    assert np.allclose(normal_equations(X, y), [1, 2, -3])
    assert np.allclose(ridge(X, y, 0.0), [1, 2, -3])


def test_descent_variants_reach_the_minimum():
    cost, grad = quadratic(A, b)
    for step, kw in [
        ("fixed", {"alpha": 0.17}),
        ("exact", {"A": A}),
        ("quadratic", {}),
    ]:
        theta, hist = gradient_descent(
            cost, grad, np.ones(2), step=step, tol=1e-8, **kw
        )
        assert np.allclose(theta, THETA_STAR, atol=1e-4), step
        assert hist.shape[1] == 3


def test_fixed_step_diverges_when_alpha_too_large():
    cost, grad = quadratic(A, b)
    theta, _ = gradient_descent(cost, grad, np.ones(2), alpha=0.21, max_iter=200)
    assert not np.allclose(theta, THETA_STAR, atol=1e-2)


def test_conjugate_gradient_n_steps():
    M = rng.normal(size=(5, 5))
    S = M @ M.T + 5 * np.eye(5)
    rhs = rng.normal(size=5)
    x, path = conjugate_gradient(S, rhs)
    assert np.allclose(S @ x, rhs) and len(path) <= 6


def test_logistic_gradient_matches_numerical():
    X = add_bias(rng.normal(size=(30, 2)))
    y = (rng.random(30) > 0.5).astype(float)
    t = rng.normal(size=3)
    eps = 1e-6
    num = [
        (
            cross_entropy_cost(t + eps * e, X, y, 1.0)
            - cross_entropy_cost(t - eps * e, X, y, 1.0)
        )
        / (2 * eps)
        for e in np.eye(3)
    ]
    assert np.allclose(cross_entropy_gradient(t, X, y, 1.0), num, atol=1e-6)


def test_logistic_and_one_vs_all_separate_blobs():
    centers = np.array([[0, 0], [4, 0], [0, 4]])
    y = np.repeat([0, 1, 2], 40)
    Xr = centers[y] + rng.normal(scale=0.5, size=(120, 2))
    X = add_bias(Xr)
    theta, _ = fit_logistic(X, (y == 1).astype(float), alpha=0.5)
    assert metrics.accuracy(y == 1, X @ theta > 0) > 0.95
    assert (
        metrics.accuracy(y, one_vs_all_predict(one_vs_all_fit(X, y, alpha=0.5), X))
        > 0.95
    )


def test_metrics_match_sklearn():
    y = rng.integers(0, 2, 200)
    s = np.clip(y * 0.3 + rng.random(200), 0, 1)
    p = (s > 0.5).astype(int)
    assert np.array_equal(metrics.confusion_matrix(y, p), skm.confusion_matrix(y, p))
    assert np.isclose(metrics.precision(y, p), skm.precision_score(y, p))
    assert np.isclose(metrics.recall(y, p), skm.recall_score(y, p))
    assert np.isclose(metrics.f1(y, p), skm.f1_score(y, p))
    assert np.isclose(
        metrics.auc(*metrics.roc_curve(y, s)[:2]), skm.roc_auc_score(y, s)
    )


def test_forward_xor_network():
    # OR and NAND in the hidden layer, AND at the output -> XOR (book ch08)
    t1 = np.array([[-10.0, 30.0], [20.0, -20.0], [20.0, -20.0]])
    t2 = np.array([[-30.0], [20.0], [20.0]])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    out = forward(X, [t1, t2])[-1].ravel()
    assert np.array_equal(out.round(), [0, 1, 1, 0])


def test_scaling_and_svd_helpers():
    X = rng.normal(5, 3, size=(100, 3))
    Z, transform = standardize(X)
    assert (
        np.allclose(Z.mean(0), 0)
        and np.allclose(Z.std(0), 1)
        and np.allclose(transform(X), Z)
    )
    assert np.allclose(low_rank_approx(X, 3), X)
    assert np.isclose(energy(np.linalg.svd(X, compute_uv=False))[-1], 1.0)
