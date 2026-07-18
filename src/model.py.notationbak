"""
src/model.py
─────────────
Student network architecture: one hidden layer, tanh activation.

    f(x; w) = W2 @ tanh(W1 @ x + b1) + b2

`w` is the single flattened decision-variable vector handed to the NLP
solver. This module is the only place that knows how `w` slices into
(W1, b1, W2, b2) -- both the CasADi-symbolic path (used inside the NLP)
and the NumPy path (used for evaluation/plotting after solving) go
through the same slicing convention so a solved `w` means the same
network in both.
"""

import numpy as np
import casadi as ca


def param_shapes(d_in, H, d_out):
    return dict(d_in=d_in, H=H, d_out=d_out)


def n_params(shapes):
    H, d_in, d_out = shapes['H'], shapes['d_in'], shapes['d_out']
    return H * d_in + H + d_out * H + d_out


def flatten(W1, b1, W2, b2):
    """Inverse of unflatten_numpy -- mainly used by tests."""
    return np.concatenate([
        W1.flatten(order='F'), b1.flatten(order='F'),
        W2.flatten(order='F'), b2.flatten(order='F'),
    ])


def unflatten_numpy(w, shapes):
    H, d_in, d_out = shapes['H'], shapes['d_in'], shapes['d_out']
    w = np.asarray(w, dtype=float).flatten()
    i = 0
    W1 = w[i:i + H * d_in].reshape(H, d_in, order='F'); i += H * d_in
    b1 = w[i:i + H].reshape(H, 1, order='F'); i += H
    W2 = w[i:i + d_out * H].reshape(d_out, H, order='F'); i += d_out * H
    b2 = w[i:i + d_out].reshape(d_out, 1, order='F'); i += d_out
    return W1, b1, W2, b2


def unflatten_symbolic(w, shapes):
    """Same slicing as unflatten_numpy, for a CasADi MX/SX vector `w`."""
    H, d_in, d_out = shapes['H'], shapes['d_in'], shapes['d_out']
    i = 0
    W1 = ca.reshape(w[i:i + H * d_in], H, d_in); i += H * d_in
    b1 = w[i:i + H]; i += H
    W2 = ca.reshape(w[i:i + d_out * H], d_out, H); i += d_out * H
    b2 = w[i:i + d_out]; i += d_out
    return W1, b1, W2, b2


def forward_symbolic(w, X, shapes):
    """X: numpy array (d_in, N). Returns a CasADi MX (d_out, N) of predictions."""
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    N = X.shape[1]
    Z1 = W1 @ ca.MX(X) + ca.repmat(b1, 1, N)
    A1 = ca.tanh(Z1)
    return W2 @ A1 + ca.repmat(b2, 1, N)


def forward_numpy(w, X, shapes):
    W1, b1, W2, b2 = unflatten_numpy(w, shapes)
    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    return W2 @ A1 + b2


def mse_numpy(w, X, y, shapes):
    yhat = forward_numpy(w, X, shapes)
    return float(np.mean((yhat - y) ** 2))


def random_init(shapes, scale=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, size=n_params(shapes))
