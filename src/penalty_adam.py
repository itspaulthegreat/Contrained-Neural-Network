"""
src/penalty_adam.py
Penalty-method baseline for the Lipschitz constraint.

Instead of an NLP solver enforcing ||W1||_F^2 ||W2||_F^2 <= L_max^2
exactly, fold it into the objective as a hinge penalty and minimize
with plain Adam:

    minimize_w   f(w) + rho * max(0, ||W1||_F^2 ||W2||_F^2 - L_max^2)

This is what the constraint looks like if you train the network the
"normal" ML way -- no NLP solver, just an extra term in the loss.
Comparing this against IPOPT's hard-constraint solve (zero violation by
construction, see src/kkt.py / src/nlp_builder.py) is the classic
penalty-method-vs-exact-constraint question: a penalty only
approximately enforces the constraint, and how well it does depends
entirely on rho.
"""

import time
import numpy as np
import casadi as ca

from src.model import (n_params, param_shapes, unflatten_symbolic,
                        forward_symbolic, mse_numpy, random_init)
from src.analysis import lipschitz_estimate


def _build_penalty_grad_fn(shapes, X_train, y_train, L_max, rho):
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)

    yhat = forward_symbolic(w, X_train, shapes)
    mse = ca.sumsqr(yhat - y_train) / X_train.shape[1]

    lip_sq = ca.sumsqr(W1) * ca.sumsqr(W2)
    violation = ca.fmax(0.0, lip_sq - L_max ** 2)
    f = mse + rho * violation

    grad = ca.gradient(f, w)
    f_fn = ca.Function('f', [w], [f])
    g_fn = ca.Function('g', [w], [grad])
    return f_fn, g_fn


def penalty_adam_optimize(w0, shapes, X_train, y_train, L_max, rho,
                           lr=0.02, n_iter=3000, beta1=0.9, beta2=0.999,
                           eps=1e-8, tol=1e-9, record_weights=False):
    f_fn, g_fn = _build_penalty_grad_fn(shapes, X_train, y_train, L_max, rho)

    w = np.asarray(w0, dtype=float).flatten()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []
    # optional per-iteration snapshots, so an animation replays THIS run
    w_history = [w.copy()] if record_weights else None

    t0 = time.time()
    n_used = n_iter
    for t in range(1, n_iter + 1):
        grad = np.asarray(g_fn(w)).flatten()
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        if record_weights:
            w_history.append(w.copy())

        loss = float(f_fn(w))
        history.append(loss)
        if t > 1 and abs(history[-2] - loss) < tol:
            n_used = t
            break
    solve_time = time.time() - t0

    return dict(w=w, history=history, n_iter=n_used, solve_time=solve_time,
                w_history=w_history)


def multi_penalty_adam_optimize(w0, shapes, X_train, y_train, rho,
                                 L_max=None, B_max=None, symmetry=False,
                                 lr=0.02, n_iter=3000, beta1=0.9, beta2=0.999,
                                 eps=1e-8, tol=0.0, record_weights=False):
    """The SOFT analogue of the FULL constraint set, so 'penalty vs hard
    constraint' can be compared like with like:

        min  MSE(w) + rho * [ max(0, ||W1||^2||W2||^2 - L^2)          (Lipschitz)
                            + max(0, ||w||^2 - B^2)                   (norm ball)
                            + sum_j max(0, b1[j] - b1[j+1]) ]         (symmetry)

    One shared rho (selected on validation like every other hyper-parameter).
    Using three separate weights would be a 3-D search -- itself an argument
    for hard constraints, and noted as such in the write-up.
    """
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    mse = ca.sumsqr(forward_symbolic(w, X_train, shapes) - y_train) / X_train.shape[1]

    pen = 0
    if L_max is not None:
        pen = pen + ca.fmax(0.0, ca.sumsqr(W1) * ca.sumsqr(W2) - L_max ** 2)
    if B_max is not None:
        pen = pen + ca.fmax(0.0, ca.sumsqr(w) - B_max ** 2)
    if symmetry:
        H = shapes['H']
        for j in range(H - 1):
            pen = pen + ca.fmax(0.0, b1[j] - b1[j + 1])

    f = mse + rho * pen
    f_fn = ca.Function('f', [w], [f])
    g_fn = ca.Function('g', [w], [ca.gradient(f, w)])

    wv = np.asarray(w0, dtype=float).flatten()
    m = np.zeros_like(wv)
    v = np.zeros_like(wv)
    history = []
    w_history = [wv.copy()] if record_weights else None
    t0 = time.time()
    n_used = n_iter
    for t in range(1, n_iter + 1):
        grad = np.asarray(g_fn(wv)).flatten()
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        wv = wv - lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
        if record_weights:
            w_history.append(wv.copy())
        loss = float(f_fn(wv))
        history.append(loss)
        if tol > 0 and t > 1 and abs(history[-2] - loss) < tol:
            n_used = t
            break
    return dict(w=wv, history=history, n_iter=n_used,
                solve_time=time.time() - t0, w_history=w_history)


def solve_penalty_adam(exp, X_train, y_train, X_test, y_test):
    shapes = param_shapes(exp['d_in'], exp['H'], exp['d_out'])
    w0 = random_init(shapes, scale=exp.get('init_scale', 0.5), seed=exp.get('seed', 0))

    # tol=0.0 disables the early stop so the run uses its full fixed budget;
    # stopping early acts as implicit regularization and would bias the rho sweep.
    opts = dict(exp['adam_opts'])
    opts.setdefault('tol', 0.0)
    out = penalty_adam_optimize(w0, shapes, X_train, y_train,
                                 L_max=exp['L_max'], rho=exp['rho'], **opts)
    w_opt = out['w']

    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)
    violation = max(0.0, lip_val ** 2 - exp['L_max'] ** 2)

    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method=exp['method'],
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=exp.get('noise_std'), rho=exp.get('rho'),
        n_vars=n_params(shapes), n_constraints=1,
        success=True, return_status='penalty_adam_complete',
        solve_time=out['solve_time'], n_iter=out['n_iter'],
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=violation,
        w=np.asarray(w_opt).tolist(), history=out['history'],
    )
