# -*- coding: utf-8 -*-
"""Pendulum, fixed spec L=4 / B=6 (Lipschitz + norm-ball). Four methods, the
gradient ones run to CONVERGENCE (loss plateau, generous cap):

  - IPOPT (hard, Lip+ball)   : the constrained solve
  - penalty-Adam (soft)      : Adam on MSE + rho*(Lipschitz + ball hinges); rho
                               swept, the best-compliant one reported with count
  - AdamW (weight decay)     : Adam + wd swept, count reported
  - plain Adam               : Adam, no constraint

Reports each method's rate / ||w|| / test MSE, iterations, wall time, and for
the two Adam-with-a-knob methods how many swept settings complied.

    python -m experiments.pendulum_convergence
        -> results/pendulum_convergence.json
        -> results/pendulum_traj.npz
"""
import json
import os
import sys
import time

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (param_shapes, n_params, forward_symbolic, unflatten_symbolic,
                       unflatten_numpy, random_init, mse_numpy)
from src.analysis import lipschitz_estimate
from src.constraints import lipschitz_constraint
from src.penalty_adam import multi_penalty_adam_optimize
from src.baseline_adam import adam_optimize
from src.callbacks import IterRecorder
from experiments import pendulum_protocol as pp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, B = 4.0, 6.0
H = pp.H
RHO_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
WD_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
TOL, CAP = 1e-9, 40000
TOLC = 1e-6

shapes = param_shapes(1, H, 1)
(Xtr, ytr), (Xva, yva), (Xte, yte) = pp.three_way_split_pendulum()
w0 = random_init(shapes, scale=0.5, seed=pp.SEED)
te = lambda w: mse_numpy(w, Xte, yte, shapes)
va = lambda w: mse_numpy(w, Xva, yva, shapes)


def checks(w):
    _, b1, _, _ = unflatten_numpy(w, shapes)
    rate = lipschitz_estimate(w, shapes); wn = float(np.linalg.norm(w))
    sym = float(np.diff(b1.flatten()).min())
    return dict(rate=rate, wnorm=wn, sym=sym, test=te(w),
                lip_ok=bool(rate <= L + TOLC), ball_ok=bool(wn <= B + TOLC),
                all_ok=bool(rate <= L + TOLC and wn <= B + TOLC))


# ---- IPOPT hard (Lipschitz + norm-ball), record iterates ----
w = ca.MX.sym("w", n_params(shapes))
f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
g, lb, ub = lipschitz_constraint(W1, W2, L)
gs = [g, ca.sumsqr(w)]
lbs = [float(lb), -ca.inf]
ubs = [float(ub), B ** 2]
rec = IterRecorder("rec_h", n_params(shapes), len(gs))
sh = ca.nlpsol("hard", "ipopt", {"x": w, "f": f, "g": ca.vertcat(*gs)},
               dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                    print_time=False, iteration_callback=rec))
t0 = time.time(); r = sh(x0=w0, lbg=np.array(lbs), ubg=np.array(ubs)); t_ip = time.time() - t0
w_ip = np.asarray(r["x"]).flatten(); it_ip = int(sh.stats()["iter_count"])
traj_ip = np.array(rec.iterates) if rec.iterates else np.array([w_ip])


# ---- penalty-Adam: sweep rho to convergence, count, pick best-compliant ----
n_pen = 0; best = None
print("penalty-Adam sweep to convergence:")
for rho in RHO_GRID:
    out = multi_penalty_adam_optimize(w0, shapes, Xtr, ytr, rho=rho, L_max=L, B_max=B,
                                      symmetry=False, n_iter=CAP, tol=TOL)
    c = checks(out["w"]); n_pen += c["all_ok"]
    print(f"  rho={rho:<7g} rate={c['rate']:.3f} |w|={c['wnorm']:.2f} sym={c['sym']:+.4f}"
          f" test={c['test']:.4f}  {'COMPLIES' if c['all_ok'] else '-'}")
    if c["all_ok"] and (best is None or va(out["w"]) < best[0]):
        best = (va(out["w"]), rho)
rho_star = best[1] if best else RHO_GRID[-1]
out = multi_penalty_adam_optimize(w0, shapes, Xtr, ytr, rho=rho_star, L_max=L, B_max=B,
                                  symmetry=False, n_iter=CAP, tol=TOL, record_weights=True)
w_pen, it_pen, t_pen, traj_pen = out["w"], int(out["n_iter"]), out["solve_time"], np.array(out["w_history"])

# ---- AdamW: sweep wd to convergence, count (never orders), pick best-val ----
n_aw = 0; best_aw = None
for wd in WD_GRID:
    o = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=CAP, weight_decay=wd, tol=TOL)
    c = checks(o["w"]); n_aw += c["all_ok"]
    if best_aw is None or va(o["w"]) < best_aw[0]:
        best_aw = (va(o["w"]), wd)
wd_star = best_aw[1]
o = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=CAP, weight_decay=wd_star, tol=TOL, record_weights=True)
w_aw, it_aw, t_aw, traj_aw = o["w"], int(o["n_iter"]), o["solve_time"], np.array(o["w_history"])

# ---- plain Adam ----
o = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=CAP, weight_decay=0.0, tol=TOL, record_weights=True)
w_pl, it_pl, t_pl, traj_pl = o["w"], int(o["n_iter"]), o["solve_time"], np.array(o["w_history"])

rows = [
    ("IPOPT (hard, Lip+ball)", w_ip, it_ip, t_ip, "no knob", "always"),
    (f"penalty-Adam (rho*={rho_star:g})", w_pen, it_pen, t_pen, f"rho* = {rho_star:g}", f"{n_pen}/{len(RHO_GRID)} rho"),
    (f"AdamW (wd*={wd_star:g})", w_aw, it_aw, t_aw, f"wd* = {wd_star:g}", f"{n_aw}/{len(WD_GRID)} wd"),
    ("plain Adam", w_pl, it_pl, t_pl, "no knob", "0/1"),
]
result = dict(L=L, B=B, sigma=pp.SIGMA, rho_star=rho_star, wd_star=wd_star,
              n_pen=n_pen, n_rho=len(RHO_GRID), n_aw=n_aw, n_wd=len(WD_GRID),
              cap=CAP, rows=[])
print(f"\n{'method':<26}{'rate':>7}{'|w|':>7}{'sym':>9}{'test':>9}{'iters':>8}{'time':>8}  complied")
for name, wv, it, dt, knob, comp in rows:
    c = checks(wv); c.update(method=name, iters=it, time=dt, knob=knob, complied=comp)
    result["rows"].append(c)
    capd = "*" if it >= CAP else ""
    print(f"{name:<26}{c['rate']:>7.3f}{c['wnorm']:>7.3f}{c['sym']:>+9.4f}"
          f"{c['test']:>9.4f}{it:>7}{capd}{dt:>8.2f}  {comp}")
json.dump(result, open(os.path.join(HERE, "results", "pendulum_convergence.json"), "w"),
          indent=1, default=float)
np.savez(os.path.join(HERE, "results", "pendulum_traj.npz"),
         ip=traj_ip, pen=traj_pen, aw=traj_aw, pl=traj_pl,
         it_ip=it_ip, it_pen=it_pen, it_aw=it_aw, it_pl=it_pl)
print("\nwrote results/pendulum_convergence.json and results/pendulum_traj.npz")
