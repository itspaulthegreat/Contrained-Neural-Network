# -*- coding: utf-8 -*-
"""
requirement_protocol.py
───────────────────────
The FAIR contest, at the headline experiment's conditions.

Slide 5 has a methodological weakness: L* = 0.15 is the argmin of a grid of
IPOPT solves, and IPOPT solving at L = 0.15 achieves exactly 0.1500 because the
constraint binds. So "IPOPT keeps it / Adam breaks it" is partly definitional --
Adam was never told the number it is being judged against.

This script removes that circularity, the way slide 42 does for the pendulum but
with all three requirements at once and with a COMPLIANCE-FIRST selection rule:

  1. The requirement is fixed UP FRONT and comes from no solver run:
        rate  ||W1||_F ||W2||_F <= 1.0   (output moves at most 1 per unit input)
        size  ||w||_2           <= 4.0   (deployment/capacity budget)
        order b1_j <= b1_{j+1}           (the symmetry-breaking spec)
     Both numerical values are round numbers set BELOW what unconstrained
     training reaches here (rate 9.57, ||w|| 10.91), so all three genuinely bind.

  2. Every method gets its own hyper-parameter sweep -- the same freedom to
     TRY to meet the spec.

  3. Selection is compliance-first: among the settings whose SOLUTION satisfies
     all three requirements, keep the one with the lowest VALIDATION MSE. If no
     setting complies, the method is reported as unable to comply.
     (Constraint satisfaction is a property of the weights, not of any data
     split, so checking it leaks nothing from the test set.)

  4. One honest TEST scoring at the end, for the compliant.

    python requirement_protocol.py  ->  results/protocol_requirement.json
"""

import json
import os

import numpy as np

from src.model import param_shapes, mse_numpy, unflatten_numpy
from src.analysis import lipschitz_estimate
from src.baseline_adam import adam_optimize
from src.penalty_adam import multi_penalty_adam_optimize
from validation_protocol import three_way_split, fit_general, H, SEED, SIGMA
from src.model import random_init

HERE = os.path.dirname(__file__)

# ── the requirement, fixed in advance (NOT read off any solver) ──────────────
L_REQ = 1.0
B_REQ = 4.0

RHO_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
WD_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
N_ITER = 3000
TOL_OK = 1e-6


def _check(w, shapes):
    """Does this weight vector satisfy all three requirements?"""
    _, b1, _, _ = unflatten_numpy(w, shapes)
    rate = lipschitz_estimate(w, shapes)
    wn = float(np.linalg.norm(w))
    worst = float(np.diff(b1.flatten()).min())
    return dict(rate=rate, wnorm=wn, sym_worst=worst,
                lip_ok=bool(rate <= L_REQ + TOL_OK),
                ball_ok=bool(wn <= B_REQ + TOL_OK),
                sym_ok=bool(worst >= -TOL_OK))


def _row(name, w, shapes, Xva, yva, Xte, yte, knob, tried, iters, time_s):
    c = _check(w, shapes)
    c.update(method=name, knob=knob, tried=tried, iters=iters, time=time_s,
             val=mse_numpy(w, Xva, yva, shapes),
             test=mse_numpy(w, Xte, yte, shapes),
             complies=bool(c['lip_ok'] and c['ball_ok'] and c['sym_ok']),
             w=w.tolist())
    return c


def run_at(sigma=SIGMA, verbose=True):
    """The whole compliance-first contest at ONE noise level."""
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split(sigma=sigma)
    w0 = random_init(shapes, scale=0.5, seed=SEED)
    rows = []

    print(f"requirement fixed up front:  rate <= {L_REQ:g},  ||w|| <= {B_REQ:g},  ordered biases")
    print(f"conditions: sigma = {sigma}, seed = {SEED}, 40/40/40, H = {H}\n")

    # ── 1. IPOPT: the requirement IS the constraint set. No sweep at all. ────
    import time as _t
    t0 = _t.time()
    w_ip = fit_general(Xtr, ytr, L=L_REQ, B=B_REQ, sym=True)
    t_ip = _t.time() - t0
    rows.append(_row('IPOPT — all three HARD', w_ip, shapes, Xva, yva, Xte, yte,
                     knob='none needed', tried=1, iters=None, time_s=t_ip))

    # ── 2. penalty-Adam: sweep rho, keep the COMPLIANT one with best val ────
    best, tried = None, 0
    for rho in RHO_GRID:
        out = multi_penalty_adam_optimize(w0, shapes, Xtr, ytr, rho=rho, L_max=L_REQ,
                                          B_max=B_REQ, symmetry=True, n_iter=N_ITER)
        tried += 1
        c = _check(out['w'], shapes)
        v = mse_numpy(out['w'], Xva, yva, shapes)
        ok = c['lip_ok'] and c['ball_ok'] and c['sym_ok']
        if verbose:
            print(f"  penalty-Adam rho={rho:<8g} rate={c['rate']:<9.4f} |w|={c['wnorm']:<8.3f} "
              f"sym={c['sym_worst']:<9.4f} val={v:.5f}  {'COMPLIES' if ok else '-'}")
        if ok and (best is None or v < best[0]):
            best = (v, rho, out['w'], out['n_iter'], out['solve_time'])
    if best:
        rows.append(_row(f'penalty-Adam (rho*={best[1]:g}, compliance-first)', best[2], shapes,
                         Xva, yva, Xte, yte, knob=f'rho = {best[1]:g}', tried=tried,
                         iters=best[3], time_s=best[4]))
    else:
        rows.append(dict(method='penalty-Adam', complies=False, knob='no rho complies',
                         tried=tried, note='no setting in the sweep satisfies all three'))

    # ── 3. AdamW: sweep weight decay, same compliance-first rule ────────────
    best, tried = None, 0
    for wd in WD_GRID:
        out = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=N_ITER,
                            weight_decay=wd, tol=0.0)
        tried += 1
        c = _check(out['w'], shapes)
        v = mse_numpy(out['w'], Xva, yva, shapes)
        ok = c['lip_ok'] and c['ball_ok'] and c['sym_ok']
        if verbose:
            print(f"  AdamW        wd={wd:<9g} rate={c['rate']:<9.4f} |w|={c['wnorm']:<8.3f} "
              f"sym={c['sym_worst']:<9.4f} val={v:.5f}  {'COMPLIES' if ok else '-'}")
        if ok and (best is None or v < best[0]):
            best = (v, wd, out['w'], out['n_iter'], out['solve_time'])
    if best:
        rows.append(_row(f'AdamW (wd*={best[1]:g}, compliance-first)', best[2], shapes,
                         Xva, yva, Xte, yte, knob=f'wd = {best[1]:g}', tried=tried,
                         iters=best[3], time_s=best[4]))
    else:
        rows.append(dict(method='AdamW', complies=False, knob='no wd complies', tried=tried,
                         note='no setting in the sweep satisfies all three'))

    # ── 4. plain Adam: no knob to turn ──────────────────────────────────────
    out = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=N_ITER,
                        weight_decay=0.0, tol=0.0)
    rows.append(_row('plain Adam (no knob)', out['w'], shapes, Xva, yva, Xte, yte,
                     knob='none available', tried=1, iters=out['n_iter'],
                     time_s=out['solve_time']))

    out = dict(L_req=L_REQ, B_req=B_REQ, sigma=sigma, seed=SEED, H=H,
               n_iter=N_ITER, rho_grid=RHO_GRID, wd_grid=WD_GRID, rows=rows)
    tag = '' if abs(sigma - SIGMA) < 1e-12 else '_sigma' + str(sigma).replace('.', 'p')
    path = os.path.join(HERE, 'results', 'protocol_requirement%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)

    print(f"\n{'method':<46}{'complies':>10}{'rate':>9}{'||w||':>9}{'test':>10}{'tried':>7}")
    for r in rows:
        if not r.get('complies') and 'rate' not in r:
            print(f"{r['method']:<46}{'NO':>10}{'-':>9}{'-':>9}{'-':>10}{r['tried']:>7}")
        else:
            print(f"{r['method']:<46}{('YES' if r['complies'] else 'no'):>10}"
                  f"{r['rate']:>9.4f}{r['wnorm']:>9.3f}{r['test']:>10.4f}{r['tried']:>7}")
    print(f"\nwrote {path}")
    return out


def main():
    return run_at(SIGMA)


def noise_sweep(sigmas=(0.1, 0.2, 0.3)):
    """Does the slide-6 conclusion depend on the noise level it was run at?
    Reruns the ENTIRE compliance-first contest at each sigma."""
    summary = []
    for sg in sigmas:
        print('=' * 78)
        summary.append((sg, run_at(sg, verbose=False)))
    print('=' * 78)
    print('NOISE SWEEP -- does the conclusion hold?\n')
    print(f"{'sigma':>6}  {'method':<36}{'complies':>9}{'rate':>9}{'test':>9}")
    for sg, out in summary:
        for r in out['rows']:
            nm = r['method'][:36]
            if 'rate' not in r:
                print(f"{sg:>6}  {nm:<36}{'NO':>9}{'-':>9}{'-':>9}")
            else:
                print(f"{sg:>6}  {nm:<36}{('YES' if r['complies'] else 'no'):>9}"
                      f"{r['rate']:>9.4f}{r['test']:>9.4f}")
        print()
    return summary


if __name__ == '__main__':
    import sys
    if '--sweep' in sys.argv:
        noise_sweep()
    else:
        main()
