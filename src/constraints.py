"""Constraint builders for the NLP. Each returns (g_expr, lb, ub) in the
format casadi.nlpsol expects for g, lbg, ubg."""

import numpy as np
import casadi as ca


def lipschitz_constraint(W1, W2, L_max):
    """Upper-bounds the network's Lipschitz constant via the certificate
    ||W1||_F^2 ||W2||_F^2 <= L_max^2 (tanh is 1-Lipschitz; ||W||_2 <= ||W||_F).
    Degree-four (biquadratic) in the weights, hence nonconvex."""
    g = ca.sumsqr(W1) * ca.sumsqr(W2)
    return g, -np.inf, float(L_max ** 2)


def norm_ball_constraint(w, B_max):
    """Hard L2 norm-ball on all weights: ||w||_2^2 <= B_max^2."""
    g = ca.sumsqr(w)
    return g, -np.inf, float(B_max ** 2)


def _spectral_norm_sq(A, n_iter=20):
    """Squared spectral norm of A (largest eigenvalue of A^T A) via symbolic
    power iteration, so it stays a differentiable CasADi expression. Fixed
    iteration count keeps the graph static."""
    n = A.shape[1]
    M = A.T @ A
    v = ca.DM.ones(n, 1)
    v = v / ca.norm_2(v)
    for _ in range(n_iter):
        v = M @ v
        v = v / ca.norm_2(v)
    return (v.T @ M @ v) / (v.T @ v)


def spectral_norm_constraint(W1, W2, s1_max, s2_max, n_iter=20):
    """Bounds the true spectral norm of each weight matrix, sigma_max(W1)^2 <=
    s1_max^2 and sigma_max(W2)^2 <= s2_max^2, via power iteration (tighter than
    the Frobenius proxy). Returns (g, lb, ub) stacked."""
    g1 = _spectral_norm_sq(W1, n_iter)
    g2 = _spectral_norm_sq(W2, n_iter)
    g = ca.vertcat(g1, g2)
    lb = -np.inf * np.ones(2)
    ub = np.array([float(s1_max) ** 2, float(s2_max) ** 2], dtype=float)
    return g, lb, ub


def symmetry_breaking_constraints(b1):
    """Orders the hidden biases b1[0] <= ... <= b1[H-1] to break the
    unit-permutation symmetry. Returns (g, lb, ub) for H-1 linear constraints,
    or (None, None, None) if H < 2."""
    H = b1.shape[0]
    if H < 2:
        return None, None, None
    g = ca.vertcat(*[b1[i] - b1[i + 1] for i in range(H - 1)])
    lb = -np.inf * np.ones(H - 1)
    ub = np.zeros(H - 1)
    return g, lb, ub
