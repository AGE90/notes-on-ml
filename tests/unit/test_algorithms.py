import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import correlate
from sklearn import metrics as skm
from sklearn.decomposition import PCA

from ml import metrics
from ml.conv import cross_correlate
from ml.data import data_loader
from ml.dynamics import damped_pendulum, dmd, lorenz, slow_manifold_system, van_der_pol
from ml.features.scaling import standardize
from ml.linalg import energy, low_rank_approx, pca
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


def test_cross_correlate_matches_scipy():
    for shape_y, shape_x in [((20,), (3,)), ((7, 9), (3, 3)), ((6, 6, 3), (3, 3, 3))]:
        y, x = rng.normal(size=shape_y), rng.normal(size=shape_x)
        assert np.allclose(cross_correlate(x, y), correlate(y, x, mode="valid"))
        z = cross_correlate(x, y, padding=1)  # "same" size for odd 3-wide filters
        assert np.allclose(z, correlate(np.pad(y, 1), x, mode="valid"))
        every_2nd = (slice(None, None, 2),) * y.ndim
        valid = correlate(y, x, mode="valid")
        assert np.allclose(cross_correlate(x, y, stride=2), valid[every_2nd])


def test_pca_matches_sklearn():
    X = rng.normal(size=(80, 5)) @ np.diag([5, 3, 2, 1, 0.5])
    Z, V, s = pca(X, 2)
    sk = PCA(2).fit(X)
    assert np.allclose(np.abs(V.T @ sk.components_.T), np.eye(2))
    assert np.allclose(np.abs(Z), np.abs(sk.transform(X)))
    assert np.allclose((s**2 / (len(X) - 1))[:2], sk.explained_variance_)


def test_dmd_recovers_a_known_linear_map():
    Q, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    eigenvalues = np.array([0.9, 0.5, 0.2])
    D = np.column_stack(
        [Q @ (eigenvalues**k * np.ones(3)) for k in range(30)]
    )  # x_k = A^k x_0 with A = Q diag(eigenvalues) Q^T
    found, modes = dmd(D[:, :-1], D[:, 1:], 3)
    assert np.allclose(np.sort(found.real), np.sort(eigenvalues))
    assert np.allclose(Q @ np.diag(eigenvalues) @ Q.T @ modes, modes * found)


def test_dynamical_systems_long_term_behaviour():
    pendulum = solve_ivp(damped_pendulum, (0, 60), [1.0, 0.0], rtol=1e-8)
    assert np.allclose(pendulum.y[:, -1], 0, atol=1e-3)
    cycle = solve_ivp(van_der_pol, (0, 100), [0.1, 0.0], rtol=1e-8, max_step=0.01)
    assert abs(np.abs(cycle.y[0, -2000:]).max() - 2.0) < 0.05  # known amplitude
    c = np.sqrt(8 / 3 * 27)
    assert np.allclose(lorenz(0, np.array([c, c, 27.0])), 0)
    x1, mu, lam = 0.7, -0.1, -1.0
    x2 = lam / (lam - 2 * mu) * x1**2
    dx = slow_manifold_system(0, np.array([x1, x2]), mu, lam)
    assert np.isclose(dx[1], 2 * x2 / x1 * dx[0])  # stays on the invariant manifold


def test_load_ptbxl_ecg_downloads_once_and_decodes(tmp_path, monkeypatch):
    samples = np.arange(24, dtype="<i2")
    calls = []

    class Response:
        content = samples.tobytes()

        def raise_for_status(self):
            pass

    def get(url, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(data_loader.requests, "get", get)
    monkeypatch.setattr(data_loader, "data_external_dir", tmp_path.joinpath)
    for _ in range(2):
        ecg = data_loader.load_ptbxl_ecg("01234")
    assert len(calls) == 1 and calls[0].endswith("/records500/01000/01234_hr.dat")
    assert ecg.shape == (2, 12) and ecg[1, 0] == 0.012
