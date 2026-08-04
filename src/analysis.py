"""Post-hoc metrics from a solved weight vector: the Lipschitz certificate,
the worst constraint violation, and the objective Hessian condition number."""

import numpy as np
import casadi as ca
from src.model import unflatten_numpy, n_params, forward_symbolic


def lipschitz_estimate(w, shapes):
    """||W1||_F * ||W2||_F -- the upper bound the Lipschitz constraint enforces."""
    W1, b1, W2, b2 = unflatten_numpy(w, shapes)
    return float(np.linalg.norm(W1, 'fro') * np.linalg.norm(W2, 'fro'))


def max_constraint_violation(g, lbg, ubg):
    if len(g) == 0:
        return 0.0
    lo = np.maximum(np.asarray(lbg) - g, 0.0)
    hi = np.maximum(g - np.asarray(ubg), 0.0)
    return float(np.max(np.concatenate([lo, hi])))


def compute_condition_number(w, shapes, X_train, y_train, eps=1e-4):
    """
    Condition number of the NLP's objective Hessian at the solution `w`.

    The MSE objective's gradient is taken symbolically from CasADi, then the
    Hessian is built by central finite differences on that gradient
    (column j = [grad(w + eps e_j) - grad(w - eps e_j)] / 2eps), symmetrized,
    and its 2-norm condition number returned.

    This is an optimization-health metric: a large condition number means the
    objective is locally ill-scaled (a long, narrow valley), which is exactly
    the regime where a BFGS-based SQP needs many more iterations than IPOPT's
    exact-Hessian interior-point steps to make progress.

    `w, shapes` identify the network; `X_train, y_train` are needed because the
    Hessian depends on the data the objective is evaluated on.
    """
    n = n_params(shapes)
    wsym = ca.MX.sym('w', n)
    yhat = forward_symbolic(wsym, X_train, shapes)
    f = ca.sumsqr(yhat - y_train) / X_train.shape[1]
    grad = ca.gradient(f, wsym)
    g_fn = ca.Function('grad', [wsym], [grad])

    w = np.asarray(w, dtype=float).flatten()
    H = np.zeros((n, n))
    for j in range(n):
        wp = w.copy(); wp[j] += eps
        wm = w.copy(); wm[j] -= eps
        H[:, j] = (np.asarray(g_fn(wp)).flatten() - np.asarray(g_fn(wm)).flatten()) / (2 * eps)
    H = 0.5 * (H + H.T)
    return float(np.linalg.cond(H))
