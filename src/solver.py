"""
src/solver.py
───────────────
Unified entry point: dispatch one experiment config to the right
optimizer (IPOPT, SQP, or Adam) and return a result dict with the same
schema regardless of method, so plotting/logging code never needs to
know which solver produced a given result.
"""

import time
import numpy as np
import casadi as ca

from src.model import param_shapes, n_params, mse_numpy, random_init
from src.nlp_builder import build_nlp
from src.baseline_adam import adam_optimize
from src.analysis import lipschitz_estimate, max_constraint_violation, compute_condition_number


def _solve_with_nlpsol(nlp_data, plugin, solver_opts):
    w, f, g = nlp_data['w'], nlp_data['f'], nlp_data['g']
    nlp = {'x': w, 'f': f, 'g': g}
    solver = ca.nlpsol('solver', plugin, nlp, solver_opts)

    t0 = time.time()
    sol = solver(x0=nlp_data['w0'], lbg=nlp_data['lbg'], ubg=nlp_data['ubg'])
    solve_time = time.time() - t0

    stats = solver.stats()
    w_opt = np.asarray(sol['x']).flatten()
    g_opt = np.asarray(sol['g']).flatten() if nlp_data['n_constraints'] > 0 else np.zeros(0)

    return dict(
        w=w_opt,
        solve_time=solve_time,
        n_iter=int(stats.get('iter_count', -1)),
        success=bool(stats.get('success', False)),
        return_status=str(stats.get('return_status', 'unknown')),
        g_opt=g_opt,
    )


def solve(exp, X_train, y_train, X_test, y_test):
    """
    exp: one fully-resolved experiment dict from configs/experiments.py
         (must contain method, H, d_in, d_out, and the relevant
         constraint / solver-option keys for that method).
    """
    shapes = param_shapes(exp['d_in'], exp['H'], exp['d_out'])

    if exp['method'] in ('ipopt', 'sqp'):
        nlp_data = build_nlp(exp, X_train, y_train)
        plugin = 'ipopt' if exp['method'] == 'ipopt' else 'sqpmethod'
        opts = exp['ipopt_opts'] if exp['method'] == 'ipopt' else exp['sqp_opts']
        out = _solve_with_nlpsol(nlp_data, plugin, opts)
        w_opt = out['w']
        n_vars = nlp_data['n_vars']
        n_constraints = nlp_data['n_constraints']
        history = []
        kkt_history = []
        g_violation = max_constraint_violation(out['g_opt'], nlp_data['lbg'], nlp_data['ubg'])
        # Hessian conditioning of the objective at the solution -- an
        # optimization-health metric (see src/analysis.py). Both constrained
        # solvers get it so IPOPT and SQP can be compared on the same problem.
        hess_cond = compute_condition_number(w_opt, shapes, X_train, y_train)

    elif exp['method'] == 'adam':
        w0 = random_init(shapes, scale=exp.get('init_scale', 0.5), seed=exp.get('seed', 0))
        adam_out = adam_optimize(w0, shapes, X_train, y_train, **exp['adam_opts'])
        w_opt = adam_out['w']
        n_vars = n_params(shapes)
        n_constraints = 0
        history = adam_out['history']
        kkt_history = adam_out['grad_inf_history']
        g_violation = 0.0
        hess_cond = None
        out = dict(solve_time=adam_out['solve_time'], n_iter=adam_out['n_iter'],
                   success=True, return_status='adam_complete')

    else:
        raise ValueError(f"Unknown method: {exp['method']}")

    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)

    return dict(
        name=exp['name'], label=exp['label'], group=exp['group'], method=exp['method'],
        H=exp['H'], L_max=exp.get('L_max'), B_max=exp.get('B_max'),
        use_lipschitz=exp.get('use_lipschitz', False),
        use_norm_ball=exp.get('use_norm_ball', False),
        use_symmetry_break=exp.get('use_symmetry_break', False),
        use_spectral_norm=exp.get('use_spectral_norm', False),
        s1_max=exp.get('s1_max'), s2_max=exp.get('s2_max'),
        noise_std=exp.get('noise_std'),
        n_vars=n_vars, n_constraints=n_constraints,
        success=out['success'], return_status=out['return_status'],
        solve_time=out['solve_time'], n_iter=out['n_iter'],
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=g_violation,
        hessian_condition_number=hess_cond,
        hessian_mode=exp.get('hessian_mode'),
        w=np.asarray(w_opt).tolist(), history=history, kkt_history=kkt_history,
    )
