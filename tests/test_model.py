"""Tests for the one-hidden-layer model: weight (un)flattening, the NumPy and
CasADi forward passes agreeing, and the MSE."""
import os
import sys
import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import (param_shapes, n_params, flatten, unflatten_numpy,
                        forward_numpy, forward_symbolic, mse_numpy, random_init)


def test_flatten_unflatten_roundtrip():
    """flatten then unflatten returns the original weight matrices."""
    shapes = param_shapes(d_in=3, H=5, d_out=2)
    rng = np.random.default_rng(1)
    W1 = rng.normal(size=(5, 3)); b1 = rng.normal(size=(5, 1))
    W2 = rng.normal(size=(2, 5)); b2 = rng.normal(size=(2, 1))

    w = flatten(W1, b1, W2, b2)
    assert w.shape[0] == n_params(shapes)

    W1r, b1r, W2r, b2r = unflatten_numpy(w, shapes)
    np.testing.assert_allclose(W1, W1r)
    np.testing.assert_allclose(b1, b1r)
    np.testing.assert_allclose(W2, W2r)
    np.testing.assert_allclose(b2, b2r)


def test_numpy_and_symbolic_forward_agree():
    """The NumPy and CasADi forward passes give the same output."""
    shapes = param_shapes(d_in=1, H=4, d_out=1)
    w = random_init(shapes, scale=0.5, seed=2)
    X = np.linspace(-2, 2, 7).reshape(1, -1)

    y_numpy = forward_numpy(w, X, shapes)

    w_sym = ca.MX.sym('w', n_params(shapes))
    expr = forward_symbolic(w_sym, X, shapes)
    f = ca.Function('f', [w_sym], [expr])
    y_symbolic = np.asarray(f(w))

    np.testing.assert_allclose(y_numpy, y_symbolic, atol=1e-10)


def test_mse_zero_for_perfect_fit():
    """MSE is (near) zero when predictions equal the targets."""
    shapes = param_shapes(d_in=1, H=4, d_out=1)
    w = random_init(shapes, scale=0.5, seed=3)
    X = np.linspace(-2, 2, 7).reshape(1, -1)
    y = forward_numpy(w, X, shapes)
    assert mse_numpy(w, X, y, shapes) < 1e-12
