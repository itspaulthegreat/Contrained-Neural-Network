"""General L-hidden-layer tanh network for the depth study. Weights stored in a
flat vector, sliced column-major in the NumPy and CasADi paths. Input sensitivity
is bounded by the product of layer norms prod_i ||W_i||_F."""

import numpy as np
import casadi as ca


def deep_shapes(d_in, hidden, d_out):
    """hidden: list of hidden-layer widths, e.g. [8] (one layer) or [8, 8] (two)."""
    return dict(dims=[d_in] + list(hidden) + [d_out])


def _layer_dims(shapes):
    dims = shapes['dims']
    return list(zip(dims[1:], dims[:-1]))          # (rows, cols) = (d_i, d_{i-1}) per weight matrix


def n_params(shapes):
    return sum(r * c + r for r, c in _layer_dims(shapes))


def unflatten_numpy(w, shapes):
    w = np.asarray(w, dtype=float).flatten()
    layers, i = [], 0
    for r, c in _layer_dims(shapes):
        W = w[i:i + r * c].reshape(r, c, order='F'); i += r * c
        b = w[i:i + r].reshape(r, 1, order='F'); i += r
        layers.append((W, b))
    return layers


def unflatten_symbolic(w, shapes):
    layers, i = [], 0
    for r, c in _layer_dims(shapes):
        W = ca.reshape(w[i:i + r * c], r, c); i += r * c
        b = w[i:i + r]; i += r
        layers.append((W, b))
    return layers


def forward_numpy(w, X, shapes):
    h = X
    layers = unflatten_numpy(w, shapes)
    for k, (W, b) in enumerate(layers):
        z = W @ h + b
        h = np.tanh(z) if k < len(layers) - 1 else z
    return h


def forward_symbolic(w, X, shapes):
    N = X.shape[1]
    h = ca.MX(X)
    layers = unflatten_symbolic(w, shapes)
    for k, (W, b) in enumerate(layers):
        z = W @ h + ca.repmat(b, 1, N)
        h = ca.tanh(z) if k < len(layers) - 1 else z
    return h


def mse_numpy(w, X, y, shapes):
    return float(np.mean((forward_numpy(w, X, shapes) - y) ** 2))


def random_init(shapes, scale=0.5, seed=0):
    return np.random.default_rng(seed).normal(0, scale, size=n_params(shapes))


def lipschitz_product_symbolic(w, shapes):
    """prod_i ||W_i||_F^2 as a CasADi expression (the squared sensitivity bound)."""
    prod = 1
    for W, _ in unflatten_symbolic(w, shapes):
        prod = prod * ca.sumsqr(W)
    return prod


def lipschitz_estimate(w, shapes):
    """prod_i ||W_i||_F -- the upper bound on the network's input sensitivity."""
    out = 1.0
    for W, _ in unflatten_numpy(w, shapes):
        out *= float(np.linalg.norm(W, 'fro'))
    return out
