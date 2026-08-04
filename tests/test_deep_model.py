"""Tests for the general deep model: parameter counts, agreement with the
one-layer model, NumPy/CasADi forward passes, and the Lipschitz product."""
import numpy as np
import casadi as ca

from src import deep_model as dm
from src import model as sm
from src.analysis import lipschitz_estimate as sm_lipschitz


def test_n_params():
    """Parameter count matches the hand-computed value at several depths."""
    assert dm.n_params(dm.deep_shapes(1, [8], 1)) == 25          # 8+8+8+1
    assert dm.n_params(dm.deep_shapes(1, [8, 8], 1)) == 97       # +64+8
    assert dm.n_params(dm.deep_shapes(1, [8, 8, 8], 1)) == 169   # +64+8


def test_one_layer_matches_model():
    """The single-hidden-layer deep model must reproduce src/model.py exactly."""
    shapes = dm.deep_shapes(1, [8], 1)
    w = dm.random_init(shapes, seed=1)
    X = np.linspace(-3, 3, 20).reshape(1, -1)
    ref = sm.forward_numpy(w, X, sm.param_shapes(1, 8, 1))
    assert np.allclose(dm.forward_numpy(w, X, shapes), ref)
    assert np.isclose(dm.lipschitz_estimate(w, shapes),
                      sm_lipschitz(w, sm.param_shapes(1, 8, 1)))


def test_symbolic_equals_numpy():
    """NumPy and CasADi deep forward passes agree at several depths."""
    for hidden in ([8], [8, 8], [6, 5, 4]):
        shapes = dm.deep_shapes(1, hidden, 1)
        n = dm.n_params(shapes)
        X = np.linspace(-2, 2, 15).reshape(1, -1)
        w = ca.MX.sym('w', n)
        f = ca.Function('f', [w], [dm.forward_symbolic(w, X, shapes)])
        wv = dm.random_init(shapes, seed=3)
        assert np.allclose(np.asarray(f(wv)), dm.forward_numpy(wv, X, shapes), atol=1e-10)


def test_lipschitz_product_positive():
    """The symbolic Frobenius-product equals the squared numeric estimate."""
    shapes = dm.deep_shapes(1, [8, 8], 1)
    w = ca.MX.sym('w', dm.n_params(shapes))
    g = ca.Function('g', [w], [dm.lipschitz_product_symbolic(w, shapes)])
    wv = dm.random_init(shapes, seed=0)
    val = float(g(wv))
    prod = dm.lipschitz_estimate(wv, shapes) ** 2
    assert np.isclose(val, prod)
