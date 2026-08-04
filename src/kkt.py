"""KKT / dual-variable analysis. IPOPT returns a Lagrange multiplier per
constraint (its shadow price): lambda > 0 means active, lambda ~ 0 means slack.
solver.solve() does not expose the duals, so this module runs the same NLP and
reads sol['lam_g'] directly."""

import time
import numpy as np
import casadi as ca

from src.model import mse_numpy
from src.nlp_builder import build_nlp
from src.analysis import lipschitz_estimate, max_constraint_violation, compute_condition_number


def solve_with_dual(exp, X_train, y_train, X_test, y_test):
    """IPOPT solve that also returns the Lipschitz constraint's multiplier as
    `lam_lipschitz`. Requires the Lipschitz constraint to be the only one active,
    so it is unambiguously g[0]."""
    if not exp.get('use_lipschitz', False):
        raise ValueError("kkt_analysis experiments must have use_lipschitz=True")
    if exp.get('use_norm_ball', False) or exp.get('use_symmetry_break', False):
        raise ValueError("kkt_analysis isolates the Lipschitz constraint -- "
                          "keep use_norm_ball / use_symmetry_break off so its "
                          "dual variable is unambiguously g[0]")

    nlp_data = build_nlp(exp, X_train, y_train)
    nlp = {'x': nlp_data['w'], 'f': nlp_data['f'], 'g': nlp_data['g']}
    solver = ca.nlpsol('solver', 'ipopt', nlp, exp['ipopt_opts'])

    t0 = time.time()
    sol = solver(x0=nlp_data['w0'], lbg=nlp_data['lbg'], ubg=nlp_data['ubg'])
    solve_time = time.time() - t0

    stats = solver.stats()
    w_opt = np.asarray(sol['x']).flatten()
    g_opt = np.asarray(sol['g']).flatten()
    lam_g = np.asarray(sol['lam_g']).flatten()
    lam_lipschitz = float(lam_g[0])

    shapes = nlp_data['shapes']
    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)
    g_violation = max_constraint_violation(g_opt, nlp_data['lbg'], nlp_data['ubg'])
    hess_cond = compute_condition_number(w_opt, shapes, X_train, y_train)

    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method=exp['method'],
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=exp.get('noise_std'),
        n_vars=nlp_data['n_vars'], n_constraints=nlp_data['n_constraints'],
        success=bool(stats.get('success', False)),
        return_status=str(stats.get('return_status', 'unknown')),
        solve_time=solve_time, n_iter=int(stats.get('iter_count', -1)),
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=g_violation,
        hessian_condition_number=hess_cond,
        w=w_opt.tolist(), history=[],
        lam_lipschitz=lam_lipschitz,
    )
