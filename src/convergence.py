"""
src/convergence.py
────────────────────
Convergence-rate study: KKT residual vs. iteration.

IPOPT is a Newton-type method -- close to the solution it takes steps
based on second-order (Hessian) information, so its KKT residual drops
superlinearly in the last iterations. Adam is a first-order method with
no curvature information: its gradient norm decays slowly (sublinearly)
and stalls at a much higher level. Plotting both residual histories on
the same axes makes the difference in *convergence rate* -- not just
final MSE -- directly visible.

The KKT residual used here:
  - IPOPT: max(inf_pr, inf_du) per iteration, read from IPOPT's own
    iteration log via `solver.stats()['iterations']` (inf_pr = primal
    infeasibility = constraint violation, inf_du = dual infeasibility =
    stationarity residual of the Lagrangian). For an unconstrained run
    inf_pr is 0 and this reduces to the gradient norm.
  - Adam (unconstrained): ||grad f(w_t)||_inf, recorded each iteration
    by src/baseline_adam.py. For an unconstrained problem this IS the
    full KKT residual, so the two curves measure the same quantity.

src/solver.py's solve() does not expose IPOPT's per-iteration log, so
-- like src/kkt.py -- this module runs the identical NLP itself through
the same unmodified building blocks and only adds the stats extraction.
"""

import time
import numpy as np
import casadi as ca

from src.model import mse_numpy
from src.nlp_builder import build_nlp
from src.analysis import lipschitz_estimate, max_constraint_violation


def solve_with_iterates(exp, X_train, y_train, X_test, y_test):
    """
    Same IPOPT solve as solver.solve(exp, ...), but additionally returns
    per-iteration residual histories:
      kkt_history     : max(inf_pr, inf_du) per iteration
      inf_pr_history  : primal infeasibility per iteration
      inf_du_history  : dual infeasibility per iteration
    """
    nlp_data = build_nlp(exp, X_train, y_train)
    nlp = {'x': nlp_data['w'], 'f': nlp_data['f'], 'g': nlp_data['g']}
    solver = ca.nlpsol('solver', 'ipopt', nlp, exp['ipopt_opts'])

    t0 = time.time()
    sol = solver(x0=nlp_data['w0'], lbg=nlp_data['lbg'], ubg=nlp_data['ubg'])
    solve_time = time.time() - t0

    stats = solver.stats()
    w_opt = np.asarray(sol['x']).flatten()
    g_opt = np.asarray(sol['g']).flatten() if nlp_data['n_constraints'] > 0 else np.zeros(0)

    iters = stats.get('iterations') or {}
    inf_pr = [float(v) for v in iters.get('inf_pr', [])]
    inf_du = [float(v) for v in iters.get('inf_du', [])]
    kkt_history = [max(p, d) for p, d in zip(inf_pr, inf_du)]

    shapes = nlp_data['shapes']
    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)
    g_violation = max_constraint_violation(g_opt, nlp_data['lbg'], nlp_data['ubg'])

    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method=exp['method'],
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=exp.get('use_lipschitz', False),
        use_norm_ball=exp.get('use_norm_ball', False),
        use_symmetry_break=exp.get('use_symmetry_break', False),
        noise_std=exp.get('noise_std'),
        n_vars=nlp_data['n_vars'], n_constraints=nlp_data['n_constraints'],
        success=bool(stats.get('success', False)),
        return_status=str(stats.get('return_status', 'unknown')),
        solve_time=solve_time, n_iter=int(stats.get('iter_count', -1)),
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=g_violation,
        w=w_opt.tolist(), history=[],
        kkt_history=kkt_history,
        inf_pr_history=inf_pr, inf_du_history=inf_du,
    )
