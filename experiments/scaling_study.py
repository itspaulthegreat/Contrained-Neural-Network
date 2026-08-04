"""Matched width scaling: IPOPT (hard) vs penalty-Adam (soft) on the same
Lipschitz + norm-ball problem, over a range of hidden widths H. Records solve
time, iterations, test MSE, sensitivity and feasibility -> scaling_study.json."""
import json
import os
import sys
import time

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_dataset
from src.model import (param_shapes, n_params, forward_symbolic, unflatten_symbolic,
                      mse_numpy, random_init)
from src.constraints import lipschitz_constraint, norm_ball_constraint
from src.analysis import lipschitz_estimate
from src.penalty_adam import multi_penalty_adam_optimize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, B = 4.0, 6.0
RHO = 1.0                                # fixed penalty weight, above the threshold
TOL, CAP = 1e-9, 40000
SIZES = [4, 8, 16, 32, 64, 96, 128, 192, 256]
SIZES_PENALTY = [4, 8, 16, 32, 64]       # penalty-Adam to convergence: tractable subset
IPOPT_OPTS = dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0), print_time=False)

Xtr, ytr, Xte, yte = generate_dataset(n_train=60, n_test=40, noise_std=0.05, seed=0)


def solve_ipopt(H):
    """Hard solve at width H: minimize MSE s.t. Lipschitz certificate + norm ball."""
    shapes = param_shapes(1, H, 1)
    w = ca.MX.sym("w", n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    g1, _, u1 = lipschitz_constraint(W1, W2, L)
    g2, _, u2 = norm_ball_constraint(w, B)
    solver = ca.nlpsol("s", "ipopt", {"x": w, "f": f, "g": ca.vertcat(g1, g2)}, IPOPT_OPTS)
    t0 = time.time()
    sol = solver(x0=random_init(shapes, seed=0), lbg=[-ca.inf, -ca.inf], ubg=[u1, u2])
    dt = time.time() - t0
    w_opt = np.asarray(sol["x"]).flatten()
    return _record(w_opt, shapes, int(solver.stats()["iter_count"]), dt,
                   bool(solver.stats()["success"]))


def run_penalty(H):
    """Matched penalty-Adam at width H, same two requirements as the hard solve."""
    shapes = param_shapes(1, H, 1)
    out = multi_penalty_adam_optimize(random_init(shapes, seed=0), shapes, Xtr, ytr,
                                      rho=RHO, L_max=L, B_max=B, symmetry=False,
                                      n_iter=CAP, tol=TOL)
    return _record(out["w"], shapes, out["n_iter"], out["solve_time"], True)


def _record(w, shapes, iters, dt, ok):
    sens = lipschitz_estimate(w, shapes); wn = float(np.linalg.norm(w))
    viol = max(max(0.0, sens ** 2 - L ** 2), max(0.0, wn ** 2 - B ** 2))
    return dict(n_vars=n_params(shapes), iters=int(iters), time=float(dt),
                test=mse_numpy(w, Xte, yte, shapes), sensitivity=sens,
                wnorm=wn, violation=viol, success=bool(ok))


def main():
    """Sweep width H; record matched IPOPT vs penalty-Adam cost and accuracy."""
    rows = []
    for H in SIZES:
        r = dict(H=H, ipopt=solve_ipopt(H), penalty=run_penalty(H))
        rows.append(r)
        ip, p = r["ipopt"], r["penalty"]
        print(f"H={H:>3} n={ip['n_vars']:>3} | IPOPT t={ip['time']:>7.2f}s it={ip['iters']:>4} "
              f"test={ip['test']:.4f} viol={ip['violation']:.0e} | penalty-Adam t={p['time']:>6.2f}s "
              f"it={p['iters']:>5} test={p['test']:.4f} sens={p['sensitivity']:.2f} viol={p['violation']:.0e}")

    out = dict(L=L, B=B, rho=RHO, sizes=SIZES, rows=rows)
    path = os.path.join(HERE, "results", "scaling_study.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
