"""
src/constraint_geometry.py
Constraint-geometry / constraint-interaction study (GROUP 9).

The Lipschitz bound and the weight norm-ball are switched on together and the
norm-ball radius B_max is swept from very tight (0.5) to loose (20.0) while
L_max is held fixed. The question is which constraint actually shapes the
solution at each B_max:

  - large B_max : the norm-ball is slack, only the Lipschitz bound binds
                  (its dual variable is nonzero, the norm-ball's is ~0)
  - small B_max : the norm-ball dominates and binds (its dual variable is
                  nonzero), squeezing the weights below what Lipschitz alone
                  would require

This is read straight off IPOPT's Lagrange multipliers (`sol['lam_g']`), the
same shadow-price mechanism src/kkt.py uses -- here for *both* constraints at
once. build_nlp adds Lipschitz first, then the norm-ball, so with symmetry
breaking off the dual rows are unambiguously lam_g[0] (Lipschitz) and
lam_g[1] (norm-ball).
"""

import time
import numpy as np
import casadi as ca

from src.model import mse_numpy
from src.nlp_builder import build_nlp
from src.analysis import (lipschitz_estimate, max_constraint_violation,
                          compute_condition_number)


def solve_with_duals(exp, X_train, y_train, X_test, y_test):
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
