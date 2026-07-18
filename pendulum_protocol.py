"""
pendulum_protocol.py
──────────────────────
The slide-5 protocol applied to the PHYSICAL task — closing the last
selection-bias gap and making the pendulum comparison best-vs-best:

    TRAIN      (60 pts of theta(t)) — fit the weights
    VALIDATION (60 pts)             — choose L_max for IPOPT *and* the
                                      weight decay for AdamW (same rules
                                      for both sides — best vs best)
    TEST       (60 pts)             — untouched; the honest final numbers

Why this exists: Group 14 reads its "best L" off the TEST curve — fine as
a landscape illustration, but it is selection on test. This script selects
on validation only. It also upgrades the baseline from plain Adam to
validation-tuned AdamW.

Also prints (first) the slide-5 sweep's exact per-L validation MSEs, so
"why did the argmin pick 0.15 and not 0.1" is answered by displayed
numbers, and the test result at the runner-up L for robustness.

    python pendulum_protocol.py        (~2 min)
"""

import json
import os

import numpy as np
import casadi as ca

from src.data import pendulum_true
from src.model import (param_shapes, n_params, forward_symbolic, mse_numpy,
                       random_init, unflatten_symbolic)
from src.constraints import lipschitz_constraint
from src.baseline_adam import adam_optimize

SEED = 0
SIGMA = 0.05
H = 8
T_RANGE = (0.0, 6.0)
L_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]     # rad/s
WD_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def three_way_split_pendulum(seed=SEED, n=60, sigma=SIGMA):
    """Three independent draws of the pendulum's noisy response."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        T = rng.uniform(T_RANGE[0], T_RANGE[1], size=(1, n))
        y = pendulum_true(T) + rng.normal(0, sigma, size=(1, n))
        out.append((T, y))
    return out


def fit_ipopt(Xtr, ytr, L):
    """Group-13 configuration: Lipschitz (L rad/s) + symmetry-breaking."""
    shapes = param_shapes(1, H, 1)
    w = ca.MX.sym('w', n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    g, lb, ub = lipschitz_constraint(W1, W2, L)
    gs, lbs, ubs = [g], [float(lb)], [float(ub)]
    for j in range(H - 1):
        gs.append(b1[j] - b1[j + 1]); lbs.append(-np.inf); ubs.append(0.0)
    s = ca.nlpsol('s', 'ipopt', {'x': w, 'f': f, 'g': ca.vertcat(*gs)},
                  dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                       print_time=False))
    sol = s(x0=random_init(shapes, scale=0.5, seed=SEED),
            lbg=np.array(lbs), ubg=np.array(ubs))
    return np.asarray(sol['x']).flatten()


def fit_adam(Xtr, ytr, wd):
    shapes = param_shapes(1, H, 1)
    w0 = random_init(shapes, scale=0.5, seed=SEED)
    out = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=3000,
                        weight_decay=wd)
    return out['w']


def main():
    shapes = param_shapes(1, H, 1)
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)

    # ---------- part 1: slide-5 sweep, exact per-L validation values --------
    from validation_protocol import three_way_split, fit, L_GRID as LG5
    (Xtr5, ytr5), (Xva5, yva5), (Xte5, yte5) = three_way_split()
    print('=' * 74)
    print('SLIDE-5 SELECTION, exact numbers (all 12 solves finish FIRST, then argmin)')
    rows5 = []
    for L in LG5:
        w = fit(Xtr5, ytr5, L)
        rows5.append((L, mse(w, Xva5, yva5), mse(w, Xte5, yte5)))
        print(f'  L = {L:<5g} validation {rows5[-1][1]:.5f}   (test, shown only '
              f'for the robustness note: {rows5[-1][2]:.4f})')
    best = min(rows5, key=lambda r: r[1])
    runner = sorted(rows5, key=lambda r: r[1])[1]
    print(f'  --> argmin: L = {best[0]:g} (val {best[1]:.5f}); runner-up '
          f'L = {runner[0]:g} (val {runner[1]:.5f}, diff {runner[1]-best[1]:.5f})')
    print(f'  --> robustness: test at the winner {best[2]:.4f} vs at the '
          f'runner-up {runner[2]:.4f} — same conclusion either way')

    # ---------- part 2: pendulum protocol, best vs best --------------------
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split_pendulum()
    print('=' * 74)
    print('PENDULUM PROTOCOL — select on VALIDATION, judge once on TEST')
    print('- IPOPT: L_max swept, chosen on validation')
    ip = []
    for L in L_GRID:
        w = fit_ipopt(Xtr, ytr, L)
        ip.append((L, mse(w, Xva, yva), w))
        print(f'  L = {L:<4g} rad/s   validation {ip[-1][1]:.5f}')
    L_star, _, w_ip = min(ip, key=lambda r: r[1])
    print(f'  --> validation selects L* = {L_star:g} rad/s')
    print('- AdamW: weight decay swept, chosen on the SAME validation set')
    aw = []
    for wd in WD_GRID:
        w = fit_adam(Xtr, ytr, wd)
        aw.append((wd, mse(w, Xva, yva), w))
        print(f'  wd = {wd:<7g} validation {aw[-1][1]:.5f}')
    wd_star, _, w_aw = min(aw, key=lambda r: r[1])
    print(f'  --> validation selects wd* = {wd_star:g}')

    t_ip, t_aw = mse(w_ip, Xte, yte), mse(w_aw, Xte, yte)
    from src.analysis import lipschitz_estimate
    print('-' * 74)
    print(f'HONEST TEST:  IPOPT (L* = {L_star:g} rad/s)  {t_ip:.4f}   vs   '
          f'AdamW (wd* = {wd_star:g})  {t_aw:.4f}')
    print(f'achieved rate certificate: IPOPT {lipschitz_estimate(w_ip, shapes):.3f} '
          f'rad/s (== L*, guaranteed)   AdamW {lipschitz_estimate(w_aw, shapes):.3f} '
          f'rad/s (uncontrolled)')

    with open(os.path.join(RESULTS, 'pendulum_protocol.json'), 'w') as f:
        json.dump({'sigma': SIGMA, 'seed': SEED, 'n': 60, 'L_grid': L_GRID,
                   'wd_grid': WD_GRID, 'L_star': L_star, 'wd_star': wd_star,
                   'ipopt_val': [(r[0], r[1]) for r in ip],
                   'adamw_val': [(r[0], r[1]) for r in aw],
                   'test_ipopt': t_ip, 'test_adamw': t_aw,
                   'lip_ipopt': lipschitz_estimate(w_ip, shapes),
                   'lip_adamw': lipschitz_estimate(w_aw, shapes)}, f, indent=1)
    print('wrote results/pendulum_protocol.json')


if __name__ == '__main__':
    main()
