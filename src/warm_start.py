"""Warm-start study. When the Lipschitz bound is tightened step by step, the
previous (looser) solution is a better initial guess than a random start, so
the solver converges in fewer iterations. Compares cold starts (fixed random
w0) against warm starts (previous solution) over the same sequence of bounds."""

import time
import numpy as np
import casadi as ca

from src.data import generate_dataset
from src.model import mse_numpy
from src.nlp_builder import build_nlp
from src.analysis import (lipschitz_estimate, max_constraint_violation,
                          compute_condition_number)


def solve_ipopt_from(cfg, X_train, y_train, X_test, y_test, x0=None):
    """One IPOPT solve of cfg's NLP. If x0 is given it is the warm start;
    otherwise build_nlp's random w0 is used (cold start)."""
    nlp_data = build_nlp(cfg, X_train, y_train)
    nlp = {'x': nlp_data['w'], 'f': nlp_data['f'], 'g': nlp_data['g']}
    solver = ca.nlpsol('solver', 'ipopt', nlp, cfg['ipopt_opts'])

    x_init = nlp_data['w0'] if x0 is None else np.asarray(x0, dtype=float).flatten()

    t0 = time.time()
    sol = solver(x0=x_init, lbg=nlp_data['lbg'], ubg=nlp_data['ubg'])
    solve_time = time.time() - t0

    stats = solver.stats()
    w_opt = np.asarray(sol['x']).flatten()
    g_opt = np.asarray(sol['g']).flatten() if nlp_data['n_constraints'] > 0 else np.zeros(0)

    shapes = nlp_data['shapes']
    train_mse = mse_numpy(w_opt, X_train, y_train, shapes)
    test_mse = mse_numpy(w_opt, X_test, y_test, shapes)
    lip_val = lipschitz_estimate(w_opt, shapes)
    g_violation = max_constraint_violation(g_opt, nlp_data['lbg'], nlp_data['ubg'])
    hess_cond = compute_condition_number(w_opt, shapes, X_train, y_train)

    return dict(
        name=cfg['name'], label=cfg['label'], group=cfg['group'], method='ipopt',
        H=cfg['H'], L_max=cfg.get('L_max'), B_max=cfg.get('B_max'),
        use_lipschitz=cfg.get('use_lipschitz', False),
        use_norm_ball=cfg.get('use_norm_ball', False),
        use_symmetry_break=cfg.get('use_symmetry_break', False),
        use_spectral_norm=cfg.get('use_spectral_norm', False),
        noise_std=cfg.get('noise_std'),
        n_vars=nlp_data['n_vars'], n_constraints=nlp_data['n_constraints'],
        success=bool(stats.get('success', False)),
        return_status=str(stats.get('return_status', 'unknown')),
        solve_time=solve_time, n_iter=int(stats.get('iter_count', -1)),
        train_mse=train_mse, test_mse=test_mse,
        lipschitz_estimate=lip_val, max_constraint_violation=g_violation,
        hessian_condition_number=hess_cond,
        w=w_opt.tolist(), history=[],
    )


def run_warm_start_study(base_cfg):
    """Run the full cold-vs-warm sweep over the (descending) bound sequence and
    return the tagged result dicts."""
    L_values = base_cfg['warm_start_L_values']

    data_seed = base_cfg.get('data_seed', base_cfg['seed'])
    X_train, y_train, X_test, y_test = generate_dataset(
        n_train=base_cfg['n_train'], n_test=base_cfg['n_test'],
        noise_std=base_cfg['noise_std'], seed=data_seed,
        d_in=base_cfg['d_in'], d_out=base_cfg['d_out'],
        H_teacher=base_cfg['H_teacher'], x_range=base_cfg['x_range'],
    )

    results = []
    for strategy in ('cold', 'warm'):
        prev_w = None
        for L in L_values:
            tag = str(L).replace('.', 'p')
            cfg = dict(base_cfg)
            cfg['L_max'] = L
            cfg['name'] = f'exp_warm_start_{strategy}_L{tag}'
            cfg['label'] = f'Warm-start study - {strategy} start, L_max={L}'

            # warm start re-uses the previous (looser) solution; cold start
            # always falls back to build_nlp's fixed random w0.
            x0 = prev_w if strategy == 'warm' else None
            res = solve_ipopt_from(cfg, X_train, y_train, X_test, y_test, x0=x0)
            res['strategy'] = strategy
            results.append(res)

            prev_w = np.array(res['w'])

    return results
