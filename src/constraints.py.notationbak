"""
src/constraints.py
────────────────────
Constraint builders for the constrained-NLP formulation. Each function
returns (g_expr, lb, ub) -- a CasADi expression plus its numeric
bounds, in the format `casadi.nlpsol` expects for `g`, `lbg`, `ubg`.

These three constraints are the actual subject of the project: the
neural network is just the parametrized function they are imposed on.
"""

import numpy as np
import casadi as ca


def lipschitz_constraint(W1, W2, L_max):
    """
    Upper-bounds the network's Lipschitz constant.

    For a 1-hidden-layer tanh network, ||f(x)-f(x')|| <= ||W2||_2 * ||W1||_2 * ||x-x'||
    (tanh is 1-Lipschitz, so it does not add to the bound). We use the
    Frobenius norm as a tractable upper bound on the spectral norm:
    ||W||_2 <= ||W||_F.

    Constraint:  ||W1||_F^2 * ||W2||_F^2  <=  L_max^2

    This is bilinear in (||W1||_F^2, ||W2||_F^2) -- a product of two
    convex terms, hence a nonconvex constraint.
    """
    g = ca.sumsqr(W1) * ca.sumsqr(W2)
    return g, -np.inf, float(L_max ** 2)


def norm_ball_constraint(w, B_max):
    """
    Hard L2 norm-ball on all weights (the constrained analogue of L2
    regularization, enforced exactly rather than penalized).

    Constraint: ||w||_2^2 <= B_max^2
    """
    g = ca.sumsqr(w)
    return g, -np.inf, float(B_max ** 2)


def _spectral_norm_sq(A, n_iter=20):
    """
    Largest eigenvalue of A^T A (== the squared spectral norm of A),
    approximated by symbolic power iteration so the whole thing stays a
    differentiable CasADi expression IPOPT can take derivatives of.

    Power iteration: v <- (A^T A) v / ||(A^T A) v||  converges to the
    dominant eigenvector of A^T A; the Rayleigh quotient v^T (A^T A) v / v^T v
    is then sigma_max(A)^2. A fixed iteration count (no convergence test)
    keeps the expression graph static. Started from an all-ones vector,
    which has a generic (nonzero) overlap with the dominant eigenvector.

    For a column vector (d_in=1 -> W1 is H x 1) or a row vector
    (d_out=1 -> W2 is 1 x H) the spectral norm coincides exactly with the
    Frobenius norm, and the iteration returns it in a single step.
    """
    n = A.shape[1]
    M = A.T @ A
    v = ca.DM.ones(n, 1)
    v = v / ca.norm_2(v)
    for _ in range(n_iter):
        v = M @ v
        v = v / ca.norm_2(v)
    return (v.T @ M @ v) / (v.T @ v)


def spectral_norm_constraint(W1, W2, s1_max, s2_max, n_iter=20):
    """
    Bounds the *true* spectral norm of each weight matrix:

        ||W1||_2 <= s1_max   AND   ||W2||_2 <= s2_max

    enforced as the squared form  sigma_max(W1)^2 <= s1_max^2  (and likewise
    for W2), with sigma_max computed by symbolic power iteration rather than
    the Frobenius proxy used by lipschitz_constraint. ||.||_2 <= ||.||_F, so
    this is a strictly tighter cap on each layer's gain than the Frobenius
    bound at the same numeric budget.

    Returns (g, lb, ub) for the two constraints stacked.
    """
    g1 = _spectral_norm_sq(W1, n_iter)
    g2 = _spectral_norm_sq(W2, n_iter)
    g = ca.vertcat(g1, g2)
    lb = -np.inf * np.ones(2)
    ub = np.array([float(s1_max) ** 2, float(s2_max) ** 2], dtype=float)
    return g, lb, ub


def symmetry_breaking_constraints(b1):
    """
    Hidden units of a 1-layer network can be freely permuted (relabeled)
    without changing the function they represent -- every permutation
    is an equally good, equivalent local minimum. Forcing the hidden
    biases into ascending order removes this combinatorial symmetry
    without restricting the function class the network can represent.

        b1[0] <= b1[1] <= ... <= b1[H-1]

    Returns (g, lb, ub) for H-1 linear constraints, or (None, None, None)
    if there are fewer than 2 hidden units to order.
    """
    H = b1.shape[0]
    if H < 2:
        return None, None, None
    g = ca.vertcat(*[b1[i] - b1[i + 1] for i in range(H - 1)])
    lb = -np.inf * np.ones(H - 1)
    ub = np.zeros(H - 1)
    return g, lb, ub
