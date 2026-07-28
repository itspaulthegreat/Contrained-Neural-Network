"""
src/baseline_adam.py
Unconstrained baseline: plain Adam gradient descent on the same MSE
objective, with no constraints at all. This is the "what everyone
already does" reference point the constrained-NLP (IPOPT / SQP)
results are measured against.

The gradient is obtained from CasADi (ca.gradient) so the comparison
uses the exact same objective expression as the constrained solves --
only the algorithm and the presence of constraints differ.
"""

import time
import numpy as np
import casadi as ca

from src.model import n_params, forward_symbolic


def _build_grad_fn(shapes, X_train, y_train):
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    yhat = forward_symbolic(w, X_train, shapes)
    f = ca.sumsqr(yhat - y_train) / X_train.shape[1]
    grad = ca.gradient(f, w)
    f_fn = ca.Function('f', [w], [f])
    g_fn = ca.Function('g', [w], [grad])
    return f_fn, g_fn


def adam_optimize(w0, shapes, X_train, y_train, lr=0.02, n_iter=3000,
                   beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-9, weight_decay=0.0,
                   record_weights=False):
    """weight_decay > 0 gives decoupled weight decay (AdamW) -- the standard
    real-world regularizer, so the baseline can be `regularized Adam`, not just
    plain Adam. It shrinks weights toward zero each step but, unlike a hard
    constraint, cannot fix a chosen sensitivity level (see the seed study)."""
    f_fn, g_fn = _build_grad_fn(shapes, X_train, y_train)

    w = np.asarray(w0, dtype=float).flatten()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []
    # ||grad f(w_t)||_inf per iteration -- for an unconstrained problem this
    # is the full KKT residual, used by the convergence_rate study to compare
    # Adam's first-order decay against IPOPT's Newton-type decay.
    grad_inf_history = []
    # optional per-iteration weight snapshots, so an animation can replay the
    # EXACT run this function performs (no duplicated optimizer loop anywhere)
    w_history = [w.copy()] if record_weights else None

    t0 = time.time()
    n_used = n_iter
    for t in range(1, n_iter + 1):
        grad = np.asarray(g_fn(w)).flatten()
        grad_inf_history.append(float(np.max(np.abs(grad))))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        if weight_decay:
            w = w - lr * weight_decay * w        # decoupled (AdamW) decay

        if record_weights:
            w_history.append(w.copy())

        loss = float(f_fn(w))
        history.append(loss)
        if t > 1 and abs(history[-2] - loss) < tol:
            n_used = t
            break
    solve_time = time.time() - t0

    return dict(w=w, history=history, grad_inf_history=grad_inf_history,
                n_iter=n_used, solve_time=solve_time, w_history=w_history)
