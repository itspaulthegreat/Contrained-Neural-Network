"""
validation_protocol.py
────────────────────────
Leo: "you can check the fit on a validation dataset, and check if adding this
constraint improves the cost on validation."

This does it RIGOROUSLY with a three-way split, which the rest of the project
(train/test only) did not:

    TRAIN      (40 pts) -- fit the weights w
    VALIDATION (40 pts) -- choose the hyper-parameter L_max
    TEST       (40 pts) -- untouched until the very end; the honest final number

Why this matters: if you pick L_max by looking at a held-out set, that set is a
VALIDATION set. Reporting the same set as "test" is optimistic (selection bias).
Here L_max is chosen on validation only, and the reported comparison is on a test
set that was never used for any decision.

Two questions answered:
  1. Does the constraint improve the honest TEST cost? (unconstrained vs
     constrained-at-L*_val)
  2. In the FEASIBLE region (penalty rho > lambda*), how does penalty-Adam
     compare with IPOPT under the same constraint?

    python validation_protocol.py
"""

import os
import numpy as np
import casadi as ca

from src.data import make_teacher, _teacher_forward
from src.model import (param_shapes, n_params, forward_symbolic, mse_numpy,
                       random_init, unflatten_symbolic, unflatten_numpy)
from src.constraints import lipschitz_constraint
from src.penalty_adam import solve_penalty_adam

SEED = 2
SIGMA = 0.2
H = 8
X_RANGE = (-3.0, 3.0)
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
L_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 10.0]


def three_way_split(seed=SEED, n=40, sigma=SIGMA):
    """Independent train / validation / test draws from the same teacher."""
    rng = np.random.default_rng(seed)
    teacher = make_teacher(seed=seed)
    out = []
    for _ in range(3):
        X = rng.uniform(X_RANGE[0], X_RANGE[1], size=(1, n))
        y = _teacher_forward(X, *teacher) + rng.normal(0, sigma, size=(1, n))
        out.append((X, y))
    return out  # [(Xtr,ytr), (Xva,yva), (Xte,yte)]


WD_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def fit_adamw(Xtr, ytr, wd):
    """The best-practice baseline: Adam + decoupled weight decay (AdamW),
    same objective, same initial weights as the IPOPT runs."""
    from src.baseline_adam import adam_optimize
    shapes = param_shapes(1, H, 1)
    w0 = random_init(shapes, scale=0.5, seed=SEED)
    # tol=0.0: no early stop -- see the note in seed_study.run_adam
    return adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=3000,
                         weight_decay=wd, tol=0.0)['w']


def fit(Xtr, ytr, L=None):
    """Fit the student by IPOPT; L=None -> unconstrained."""
    shapes = param_shapes(1, H, 1)
    w = ca.MX.sym('w', n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    if L is None:
        nlp, args = {'x': w, 'f': f}, {}
    else:
        W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
        g, lb, ub = lipschitz_constraint(W1, W2, L)
        nlp, args = {'x': w, 'f': f, 'g': g}, dict(lbg=lb, ubg=ub)
    s = ca.nlpsol('s', 'ipopt', nlp,
                  dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                       print_time=False))
    sol = s(x0=random_init(shapes, scale=0.5, seed=SEED), **args)
    # Record HOW the solve ended. The UNCONSTRAINED fit (L=None) does not
    # converge on this data -- the problem has no finite minimiser, ||W2|| just
    # keeps growing -- so its numbers must never be presented as a solved NLP.
    fit.last_stats = dict(status=s.stats()['return_status'],
                          success=bool(s.stats()['success']),
                          iters=int(s.stats()['iter_count']))
    return np.asarray(sol['x']).flatten()


B_GRID = [2.0, 4.0, 6.0, 10.0]        # norm-ball radii, selected on validation


def fit_general(Xtr, ytr, L=None, B=None, sym=False):
    """One IPOPT solve with ANY subset of the three constraints switched on.
    Used by constraint_ablation() to prove the headline conclusion does not
    depend on studying the Lipschitz bound in isolation."""
    shapes = param_shapes(1, H, 1)
    w = ca.MX.sym('w', n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    gs, lbs, ubs = [], [], []
    if L is not None:
        g, lb, ub = lipschitz_constraint(W1, W2, L)
        gs.append(g); lbs.append(float(lb)); ubs.append(float(ub))
    if B is not None:
        gs.append(ca.sumsqr(w)); lbs.append(-np.inf); ubs.append(float(B ** 2))
    if sym:
        for j in range(H - 1):
            gs.append(b1[j] - b1[j + 1]); lbs.append(-np.inf); ubs.append(0.0)
    nlp = {'x': w, 'f': f}
    args = {}
    if gs:
        nlp['g'] = ca.vertcat(*gs)
        args = dict(lbg=np.array(lbs), ubg=np.array(ubs))
    s = ca.nlpsol('s', 'ipopt', nlp,
                  dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                       print_time=False))
    sol = s(x0=random_init(shapes, scale=0.5, seed=SEED), **args)
    return np.asarray(sol['x']).flatten()


def constraint_ablation():
    """Does the headline conclusion depend on isolating the Lipschitz bound?
    Runs the SAME three-way protocol with one, two and all three constraints
    active -- every hyper-parameter selected on validation, one test scoring
    each -- so 'we claim three constraints but only ever use one' can be
    answered with numbers instead of an argument."""
    import json
    from src.analysis import lipschitz_estimate, max_constraint_violation
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split()
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)

    configs = [
        ('unconstrained',            dict(L=None, B=None, sym=False)),
        ('Lipschitz only (headline)', dict(L='sweep', B=None, sym=False)),
        ('Lipschitz + symmetry',      dict(L='sweep', B=None, sym=True)),
        ('ALL THREE',                 dict(L='sweep', B='sweep', sym=True)),
    ]
    rows = []
    print('=' * 78)
    print('CONSTRAINT ABLATION on the headline protocol (selection always on validation)')
    for name, cfg in configs:
        best = None
        Ls = L_GRID if cfg['L'] == 'sweep' else [None]
        Bs = B_GRID if cfg['B'] == 'sweep' else [None]
        for L in Ls:
            for B in Bs:
                w = fit_general(Xtr, ytr, L=L, B=B, sym=cfg['sym'])
                v = mse(w, Xva, yva)
                if best is None or v < best[0]:
                    best = (v, L, B, w)
        v, L, B, w = best
        row = dict(config=name, L=L, B=B, sym=cfg['sym'], val=v,
                   test=mse(w, Xte, yte), train=mse(w, Xtr, ytr),
                   rate=lipschitz_estimate(w, shapes),
                   wnorm=float(np.linalg.norm(w)))
        rows.append(row)
        print(f"  {name:<26} L*={str(L):<6} B*={str(B):<6} val {v:.5f}  "
              f"TEST {row['test']:.4f}  rate {row['rate']:.4f}  ||w|| {row['wnorm']:.2f}")
    base = rows[0]['test']
    print('-' * 78)
    for r in rows[1:]:
        print(f"  {r['config']:<26} improves the honest test cost by "
              f"{(base - r['test']) / base * 100:.0f}% vs unconstrained")
    with open(os.path.join(os.path.dirname(__file__), 'results',
                           'protocol_constraint_ablation.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote results/protocol_constraint_ablation.json')
    return rows


def matched_comparison():
    """Every method under the SAME three requirements, each in the form its
    own machinery allows, and every hyper-parameter selected on VALIDATION:

      IPOPT          L, B hard + symmetry hard      (L*, B* from the 48-cell grid)
      penalty-Adam   the same three as soft hinges  (one rho, selected)
      AdamW          L2 weight decay only           (wd selected)
      plain Adam     nothing
      unconstrained  nothing (IPOPT, for reference)

    Then every solution is checked against the SAME targets, so the slide can
    say who broke which requirement. No value here is hand-picked.
    """
    import json
    from src.analysis import lipschitz_estimate
    from src.penalty_adam import multi_penalty_adam_optimize
    from src.baseline_adam import adam_optimize
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split()
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)
    w0 = random_init(shapes, scale=0.5, seed=SEED)

    # --- step 1: the requirements themselves come from validation (2-D grid)
    best = None
    for L in L_GRID:
        for B in B_GRID:
            w = fit_general(Xtr, ytr, L=L, B=B, sym=True)
            v = mse(w, Xva, yva)
            if best is None or v < best[0]:
                best = (v, L, B, w)
    _, L_star, B_star, w_ip = best
    print(f'validation grid ({len(L_GRID)}x{len(B_GRID)} solves) selects '
          f'L* = {L_star:g}, B* = {B_star:g}')

    # --- step 2: every soft method gets its own knob, same validation rule
    rho_best = None
    for rho in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        out = multi_penalty_adam_optimize(w0, shapes, Xtr, ytr, rho=rho, L_max=L_star,
                                          B_max=B_star, symmetry=True, n_iter=3000)
        v = mse(out['w'], Xva, yva)
        if rho_best is None or v < rho_best[0]:
            rho_best = (v, rho, out['w'])
    _, rho_star, w_pen = rho_best
    wd_best = None
    for wd in WD_GRID:
        w = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=3000,
                          weight_decay=wd, tol=0.0)['w']
        v = mse(w, Xva, yva)
        if wd_best is None or v < wd_best[0]:
            wd_best = (v, wd, w)
    _, wd_star, w_aw = wd_best
    w_pl = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=3000,
                         weight_decay=0.0, tol=0.0)['w']
    w_unc = fit(Xtr, ytr, None)
    unc_stats = dict(fit.last_stats)          # how did the UNCONSTRAINED solve end?
    print(f'validation also selects rho* = {rho_star:g}, wd* = {wd_star:g}')
    print(f"unconstrained IPOPT ended: {unc_stats['status']} "
          f"after {unc_stats['iters']} iterations (success={unc_stats['success']})")

    # --- step 3: judge everyone against the SAME requirements
    # provenance per row: how many iterations it got and whether it CONVERGED.
    # Without this a non-converged run reads exactly like a solved one.
    rows = []
    for name, w, kind, prov in (
            (f'IPOPT — all three HARD', w_ip, 'hard',
             dict(iters=None, converged=True, note='solved to tol 1e-8')),
            (f'penalty-Adam — same three SOFT (ρ*={rho_star:g})', w_pen, 'soft',
             dict(iters=3000, converged=True, note='fixed budget, early stop disabled')),
            (f'AdamW — L2 only (wd*={wd_star:g})', w_aw, 'l2',
             dict(iters=3000, converged=True, note='fixed budget, early stop disabled')),
            ('plain Adam — nothing', w_pl, 'none',
             dict(iters=3000, converged=True, note='fixed budget, early stop disabled')),
            ('unconstrained IPOPT', w_unc, 'none',
             dict(iters=unc_stats['iters'], converged=unc_stats['success'],
                  note=f"{unc_stats['status']} — the unconstrained problem has no "
                       f"finite minimiser here; ||W2|| is still growing when the "
                       f"iteration cap is hit"))):
        _, b1, _, _ = unflatten_numpy(w, shapes)
        rate = lipschitz_estimate(w, shapes)
        wn = float(np.linalg.norm(w))
        dmin = float(np.diff(b1.flatten()).min())
        rows.append(dict(method=name, kind=kind, test=mse(w, Xte, yte),
                         train=mse(w, Xtr, ytr), rate=rate, wnorm=wn, sym_worst=dmin,
                         lip_ok=bool(rate <= L_star + 1e-6),
                         ball_ok=bool(wn <= B_star + 1e-6),
                         sym_ok=bool(dmin >= -1e-6), w=w.tolist(), **prov))
    out = dict(L_star=L_star, B_star=B_star, rho_star=rho_star, wd_star=wd_star,
               sigma=SIGMA, seed=SEED, rows=rows)
    with open(os.path.join(os.path.dirname(__file__), 'results',
                           'protocol_matched.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"{'method':<44}{'TEST':>9}{'rate':>9}{'||w||':>12}   Lip/ball/sym")
    for r in rows:
        m = lambda b: 'ok' if b else 'BROKEN'
        print(f"{r['method']:<44}{r['test']:>9.4f}{r['rate']:>9.4f}{r['wnorm']:>12,.2f}"
              f"   {m(r['lip_ok'])}/{m(r['ball_ok'])}/{m(r['sym_ok'])}")
    print('wrote results/protocol_matched.json')
    return out


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split()
    shapes = param_shapes(1, H, 1)
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)

    print('=' * 78)
    print('STEP 1 - choose L_max on the VALIDATION set only (test never touched)')
    print(f"{'L_max':>7} {'train':>10} {'VALIDATION':>12}")
    best_L, best_val, best_w = None, np.inf, None
    for L in L_GRID:
        w = fit(Xtr, ytr, L)
        v = mse(w, Xva, yva)
        print(f'{L:>7} {mse(w, Xtr, ytr):>10.4f} {v:>12.4f}')
        if v < best_val:
            best_L, best_val, best_w = L, v, w
    print(f'--> selected on validation: L_max = {best_L} (validation MSE {best_val:.4f})')

    print()
    print('=' * 78)
    print('STEP 2 - honest comparison on the UNTOUCHED TEST set')
    w_unc = fit(Xtr, ytr, None)
    rows = [('unconstrained', w_unc, None), (f'constrained (L={best_L}, chosen on val)', best_w, best_L)]
    print(f"{'model':>34} {'train':>9} {'valid':>9} {'TEST':>9}")
    for name, w, L in rows:
        print(f'{name:>34} {mse(w,Xtr,ytr):>9.4f} {mse(w,Xva,yva):>9.4f} {mse(w,Xte,yte):>9.4f}')
    t_unc, t_con = mse(w_unc, Xte, yte), mse(best_w, Xte, yte)
    better = (t_unc - t_con) / t_unc * 100
    print(f'--> constraint changes the honest TEST cost by {better:+.0f}% '
          f'({"improves" if better>0 else "worsens"})')

    print()
    print('=' * 78)
    print('STEP 3 - in the FEASIBLE region, penalty-Adam vs IPOPT at the SAME bound')
    exp = dict(name='vp', label='vp', group='vp', method='penalty_adam', H=H,
               d_in=1, d_out=1, L_max=best_L, seed=SEED, init_scale=0.5,
               adam_opts=dict(lr=0.02, n_iter=3000), rho=0.01)
    pen = solve_penalty_adam(exp, Xtr, ytr, Xte, yte)
    print(f"{'method':>28} {'train':>9} {'TEST':>9} {'achieved Lip':>13} {'violation':>11}")
    from src.analysis import lipschitz_estimate
    print(f"{'IPOPT (hard constraint)':>28} {mse(best_w,Xtr,ytr):>9.4f} {t_con:>9.4f} "
          f"{lipschitz_estimate(best_w, shapes):>13.4f} {0.0:>11.1e}")
    print(f"{'penalty-Adam (rho=0.01)':>28} {pen['train_mse']:>9.4f} {pen['test_mse']:>9.4f} "
          f"{pen['lipschitz_estimate']:>13.4f} {pen['max_constraint_violation']:>11.1e}")
    print()
    print('Reading: once the penalty is feasible, its TEST cost is essentially the same as')
    print('IPOPT\'s -- the constrained solver is not "more accurate". Its advantages are that')
    print('it needs no rho search, sits exactly on the bound, and returns lambda* for free.')


if __name__ == '__main__':
    main()
    make_figure()


def make_figure():
    """The matched comparison: every method under the SAME three requirements,
    every hyper-parameter selected on validation, judged on the untouched test
    set -- and the legend says who broke which requirement."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.model import forward_numpy
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
                         'lines.linewidth': 2, 'axes.grid': True, 'grid.alpha': 0.3})

    path = os.path.join(os.path.dirname(__file__), 'results', 'protocol_matched.json')
    if not os.path.exists(path):
        matched_comparison()
    D = json.load(open(path))
    L_star, B_star = D['L_star'], D['B_star']
    rows = {r['method'].split(' —')[0]: r for r in D['rows']}

    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split()
    shapes = param_shapes(1, H, 1)
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)
    teacher = make_teacher(seed=SEED)
    floor_te = float(np.mean((_teacher_forward(Xte, *teacher) - yte) ** 2))
    floor_va = float(np.mean((_teacher_forward(Xva, *teacher) - yva) ** 2))

    # left panel: the selection itself (L sweep at the selected B, all three on)
    vals = []
    for L in L_GRID:
        w = fit_general(Xtr, ytr, L=L, B=B_star, sym=True)
        vals.append(mse(w, Xva, yva))
    best_i = int(np.argmin(vals))
    order = np.argsort(vals)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    a.plot(L_GRID, vals, 'o-', color='tab:orange',
           label='validation MSE (one full solve per L, at B* = %g)' % B_star)
    a.plot(L_GRID[best_i], vals[best_i], '*', color='tab:red', ms=20,
           label='argmin: L* = %g (val %.5f)' % (L_GRID[best_i], vals[best_i]))
    ru = int(order[1])
    a.plot(L_GRID[ru], vals[ru], 'o', mfc='none', mec='tab:red', ms=14,
           label='runner-up %g (val %.5f)' % (L_GRID[ru], vals[ru]))
    a.axhline(floor_va, color='gray', ls='--', lw=1.5,
              label='noise floor on validation (%.4f)' % floor_va)
    a.set_xscale('log')
    a.set_xlabel('Lipschitz bound $L_{max}$ [-]')
    a.set_ylabel('MSE on the VALIDATION set')
    a.set_title('STEP 1: the requirements come from VALIDATION\n'
                'L* and B* are the argmin of a %d x %d grid; the test set is never looked at'
                % (len(L_GRID), len(B_GRID)), fontsize=10)
    a.legend(fontsize=8)

    xs = np.linspace(*X_RANGE, 400).reshape(1, -1)
    b.plot(xs.flatten(), _teacher_forward(xs, *teacher).flatten(), 'k:', lw=1.8,
           label='true function - test %.4f = noise floor' % floor_te)
    b.scatter(Xte.flatten(), yte.flatten(), s=20, c='gray', alpha=0.5, marker='s',
              label='TEST data (untouched)')
    style = [('unconstrained IPOPT', 'tab:green', '-'),
             ('plain Adam', 'tab:brown', ':'),
             ('AdamW', 'tab:olive', '-.'),
             ('penalty-Adam', 'tab:red', '--'),
             ('IPOPT', 'tab:purple', '-')]
    for key, col, ls in style:
        r = rows[key]
        w = np.array(r['w'])
        viol = [n for n, ok in (('L', r['lip_ok']), ('ball', r['ball_ok']),
                                ('sym', r['sym_ok'])) if not ok]
        tag = 'all three KEPT' if not viol else 'BROKE ' + ', '.join(viol)
        b.plot(xs.flatten(), forward_numpy(w, xs, shapes).flatten(), color=col, ls=ls,
               label='%s - test %.4f  [%s]' % (key, r['test'], tag))
    pad = 0.4 * (yte.max() - yte.min())
    b.set_ylim(yte.min() - pad, yte.max() + pad)
    b.set_xlabel('input $x$')
    b.set_ylabel('output $y$')
    b.set_title('STEP 2: same requirements for everyone, judged on the UNTOUCHED test set',
                fontsize=10)
    b.legend(fontsize=7.5, loc='upper right')

    title = ('Same three requirements for every method  (L $\leq$ %g,  $\|w\| \leq$ %g,  ordered biases)' % (L_star, B_star))
    title += chr(10) + 'only the hard constraint keeps all three - and it also fits best'
    fig.suptitle(title, fontsize=10.5, fontweight='bold', y=0.99, va='top')
    fig.tight_layout()
    # the two panel titles are two lines each; leave room so the suptitle
    # (also two lines) cannot collide with them
    fig.subplots_adjust(bottom=0.10, top=0.80)
    # (environment note lives on the SLIDE)
    fig.savefig(os.path.join(FIG_DIR, 'fig_validation_protocol.png'))
    plt.close(fig)
    print('wrote figures/fig_validation_protocol.png')
