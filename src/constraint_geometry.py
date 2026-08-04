"""Constraint-interaction study. Lipschitz and norm-ball are both on while the
ball radius B_max is swept; the Lagrange multipliers (lam_g[0]=Lipschitz,
lam_g[1]=norm-ball) show which constraint binds at each radius."""

import time
import numpy as np
import casadi as ca

from src.model import mse_numpy
from src.nlp_builder import build_nlp
from src.analysis import (lipschitz_estimate, max_constraint_violation,
                          compute_condition_number)


def solve_with_duals(exp, X_train, y_train, X_test, y_test):
    """IPOPT solve returning both constraints' multipliers and which one binds."""
    if not (exp.get('use_lipschitz', False) and exp.get('use_norm_ball', False)):
        raise ValueError("constraint_geometry needs BOTH use_lipschitz and "
                         "use_norm_ball True (it studies their interaction)")
    if exp.get('use_symmetry_break', False):
        raise ValueError("constraint_geometry keeps use_symmetry_break off so "
                         "the dual rows are exactly g[0]=Lipschitz, g[1]=norm-ball")

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
    lam_norm_ball = float(lam_g[1])

    # "active" = the constraint with the larger |dual| (binding, costly).
    active = 'lipschitz' if abs(lam_lipschitz) >= abs(lam_norm_ball) else 'norm_ball'

    shapes = nlp_data['shapes']
    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)
    g_violation = max_constraint_violation(g_opt, nlp_data['lbg'], nlp_data['ubg'])
    hess_cond = compute_condition_number(w_opt, shapes, X_train, y_train)

    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method='ipopt',
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=False,
        use_spectral_norm=False,
        noise_std=exp.get('noise_std'),
        n_vars=nlp_data['n_vars'], n_constraints=nlp_data['n_constraints'],
        success=bool(stats.get('success', False)),
        return_status=str(stats.get('return_status', 'unknown')),
        solve_time=solve_time, n_iter=int(stats.get('iter_count', -1)),
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=g_violation,
        hessian_condition_number=hess_cond,
        lam_lipschitz=lam_lipschitz, lam_norm_ball=lam_norm_ball,
        active_constraint=active,
        w=w_opt.tolist(), history=[],
    )
