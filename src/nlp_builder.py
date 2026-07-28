"""
src/nlp_builder.py
Assembles the constrained NLP that is the actual subject of this project:

    decision vars : w        (flattened network weights)
    objective     : f(w)     mean-squared training error
    constraints   : g(w)     Lipschitz bound, weight norm-ball,
                              symmetry-breaking -- each independently
                              switchable via the experiment config

Everything else (the network architecture, the synthetic data) exists
only to give this NLP something nontrivial to optimize.
"""

import numpy as np
import casadi as ca

from src.model import param_shapes, n_params, unflatten_symbolic, forward_symbolic, random_init
from src.constraints import (lipschitz_constraint, norm_ball_constraint,
                             symmetry_breaking_constraints, spectral_norm_constraint)


def build_nlp(cfg, X_train, y_train):
    shapes = param_shapes(cfg['d_in'], cfg['H'], cfg['d_out'])
    n = n_params(shapes)
    w = ca.MX.sym('w', n)

    yhat = forward_symbolic(w, X_train, shapes)
    f = ca.sumsqr(yhat - y_train) / X_train.shape[1]  # MSE

    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)

    g_list, lb_list, ub_list = [], [], []

    if cfg.get('use_lipschitz', False):
        g, lb, ub = lipschitz_constraint(W1, W2, cfg['L_max'])
        g_list.append(g); lb_list.append(lb); ub_list.append(ub)

    if cfg.get('use_norm_ball', False):
        g, lb, ub = norm_ball_constraint(w, cfg['B_max'])
        g_list.append(g); lb_list.append(lb); ub_list.append(ub)

    if cfg.get('use_spectral_norm', False):
        g, lb, ub = spectral_norm_constraint(W1, W2, cfg['s1_max'], cfg['s2_max'])
        g_list.append(g); lb_list.append(lb); ub_list.append(ub)

    if cfg.get('use_symmetry_break', False):
        g, lb, ub = symmetry_breaking_constraints(b1)
        if g is not None:
            g_list.append(g); lb_list.append(lb); ub_list.append(ub)

    if g_list:
        g_expr = ca.vertcat(*g_list)
        lbg = np.concatenate([np.atleast_1d(np.asarray(x, dtype=float)) for x in lb_list])
        ubg = np.concatenate([np.atleast_1d(np.asarray(x, dtype=float)) for x in ub_list])
    else:
        g_expr = ca.MX(0, 1)
        lbg = np.zeros(0)
        ubg = np.zeros(0)

    w0 = random_init(shapes, scale=cfg.get('init_scale', 0.5), seed=cfg.get('seed', 0))

    return dict(
        w=w, f=f, g=g_expr, lbg=lbg, ubg=ubg, w0=w0,
        shapes=shapes, n_vars=n, n_constraints=int(g_expr.shape[0]),
    )
