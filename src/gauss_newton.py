"""
src/gauss_newton.py
Self-implemented Gauss-Newton / Levenberg-Marquardt method for the
unconstrained MSE fit (course notes, Numerical Methods 6.2 and 6.7).

The objective is a nonlinear least-squares problem

    f(w) = (1/N) Σᵢ (f(xᵢ; w) − yᵢ)²  =  ‖r(w)‖₂²,   rᵢ = (ŷᵢ − yᵢ)/√N,

so the Gauss-Newton idea applies directly: linearize the residual r at the
current iterate and solve the resulting *linear* least-squares problem.
That is equivalent to a Newton step with the Hessian approximation

    ∇²f ≈ 2 JᵀJ,      J = ∂r/∂w   (exact from CasADi AD),

which drops the Σ rᵢ∇²rᵢ term - exact in the zero-residual limit, and
positive semidefinite by construction.

On this problem JᵀJ is SINGULAR (overparameterization + hidden-unit
permutation symmetry -- the same degeneracy that breaks unconstrained
Newton in the convergence study), so plain Gauss-Newton division is
"numerically dangerous" (course notes, §6.6). We therefore use the
Levenberg-Marquardt regularization of §6.7:

    (JᵀJ + λ_lm I) Δw = −Jᵀr,

with the classical adaptive damping: shrink λ_lm after a successful step
(→ Gauss-Newton behaviour near the solution), grow it after a rejected
step (→ small gradient-descent-like steps far away).

The run records ‖∇f(w_k)‖_inf per iteration -- for an unconstrained
problem this is the KKT residual -- so it drops straight into the
Group-10 convergence-rate comparison next to Adam and IPOPT.
"""

import time
import numpy as np
import casadi as ca

from src.model import n_params, forward_symbolic, random_init


def _build_residual_fns(shapes, X_train, y_train):
    n = n_params(shapes)
    N = X_train.shape[1]
    w = ca.MX.sym('w', n)
    yhat = forward_symbolic(w, X_train, shapes)
    r = (yhat - y_train).T / np.sqrt(N)          # (N, 1): f = ||r||^2
    J = ca.jacobian(r, w)                         # (N, n) exact AD Jacobian
    f = ca.sumsqr(r)
    grad = ca.gradient(f, w)
    return (ca.Function('r', [w], [r]), ca.Function('J', [w], [J]),
            ca.Function('f', [w], [f]), ca.Function('g', [w], [grad]))


def gauss_newton_optimize(w0, shapes, X_train, y_train, max_iter=200,
                           lam0=1e-3, tol=1e-8):
    """Levenberg-Marquardt loop. Returns the same dict shape as
    adam_optimize: w, history (loss), grad_inf_history, n_iter, solve_time."""
    r_fn, J_fn, f_fn, g_fn = _build_residual_fns(shapes, X_train, y_train)

    w = np.asarray(w0, dtype=float).flatten()
    lam = lam0
    history, grad_inf_history = [], []
    f_cur = float(f_fn(w))

    t0 = time.time()
    n_used = max_iter
    for k in range(1, max_iter + 1):
        r = np.asarray(r_fn(w)).flatten()
        J = np.asarray(J_fn(w))
        g = 2.0 * J.T @ r                          # ∇f = 2 Jᵀr
        grad_inf_history.append(float(np.max(np.abs(g))))
        if grad_inf_history[-1] < tol:
            n_used = k - 1
            break

        # LM step: (JᵀJ + λI) Δw = −Jᵀr ; retry with larger λ until accepted
        JtJ = J.T @ J
        accepted = False
        for _ in range(30):
            try:
                dw = np.linalg.solve(JtJ + lam * np.eye(len(w)), -J.T @ r)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            f_new = float(f_fn(w + dw))
            if f_new < f_cur:                       # sufficient: plain decrease
                w = w + dw
                f_cur = f_new
                lam = max(lam / 3.0, 1e-12)         # trust the model more
                accepted = True
                break
            lam *= 10.0                             # damp harder, retry
        history.append(f_cur)
        if not accepted:                            # cannot decrease anymore
            n_used = k
            break
    solve_time = time.time() - t0

    return dict(w=w, history=history, grad_inf_history=grad_inf_history,
                n_iter=n_used, solve_time=solve_time)


def solve_gauss_newton(exp, X_train, y_train, X_test, y_test):
    """Dispatch wrapper mirroring solver.solve()'s result schema."""
    from src.model import param_shapes, mse_numpy
    from src.analysis import lipschitz_estimate

    shapes = param_shapes(exp['d_in'], exp['H'], exp['d_out'])
    w0 = random_init(shapes, scale=exp.get('init_scale', 0.5), seed=exp.get('seed', 0))
    out = gauss_newton_optimize(w0, shapes, X_train, y_train,
                                 **exp.get('gn_opts', {}))

    w_opt = out['w']
    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method=exp['method'],
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=False, use_norm_ball=False, use_symmetry_break=False,
        use_spectral_norm=False, s1_max=None, s2_max=None,
        noise_std=exp.get('noise_std'),
        n_vars=n_params(shapes), n_constraints=0,
        success=True, return_status='gauss_newton_complete',
        solve_time=out['solve_time'], n_iter=out['n_iter'],
        train_mse=mse_numpy(w_opt, X_train, y_train, shapes),
        test_mse=mse_numpy(w_opt, X_test, y_test, shapes),
        lipschitz_estimate=lipschitz_estimate(w_opt, shapes),
        max_constraint_violation=0.0,
        hessian_condition_number=None, hessian_mode='gauss-newton',
        w=np.asarray(w_opt).tolist(),
        history=out['history'], kkt_history=out['grad_inf_history'],
    )
