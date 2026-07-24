"""
Width scaling, matched: what does the interior-point solve cost as the network
grows, compared with training the SAME requirements as a soft penalty?

At each hidden width H, on the synthetic task, three methods solve the same
Lipschitz + norm-ball problem:

  - IPOPT (hard)      : the constrained NLP, solved to KKT tolerance
  - penalty-Adam      : Adam on MSE + rho*(Lipschitz + ball hinges), to convergence
  - plain Adam        : unconstrained reference (no requirements)

Recorded per method: solve time, iterations, test MSE, achieved sensitivity and
feasibility. IPOPT and plain Adam span the full range; penalty-Adam (run to
convergence) is limited to the smaller sizes where that is tractable.

    python -m experiments.scaling_study   ->  results/scaling_study.json
"""
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
from src.baseline_adam import adam_optimize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, B = 4.0, 6.0
RHO = 1.0                                # fixed penalty weight, above the threshold
TOL, CAP = 1e-9, 40000
SIZES = [4, 8, 16, 32, 64, 96, 128, 192, 256]
SIZES_PENALTY = [4, 8, 16, 32, 64]       # penalty-Adam to convergence: tractable subset
IPOPT_OPTS = dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0), print_time=False)

Xtr, ytr, Xte, yte = generate_dataset(n_train=60, n_test=40, noise_std=0.05, seed=0)


def solve_ipopt(H):
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
    shapes = param_shapes(1, H, 1)
    out = multi_penalty_adam_optimize(random_init(shapes, seed=0), shapes, Xtr, ytr,
                                      rho=RHO, L_max=L, B_max=B, symmetry=False,
                                      n_iter=CAP, tol=TOL)
    return _record(out["w"], shapes, out["n_iter"], out["solve_time"], True)


def run_adam(H):
    shapes = param_shapes(1, H, 1)
    out = adam_optimize(random_init(shapes, seed=0), shapes, Xtr, ytr, lr=0.02,
                        n_iter=CAP, weight_decay=0.0, tol=TOL)
    return _record(out["w"], shapes, out["n_iter"], out["solve_time"], True)


def _record(w, shapes, iters, dt, ok):
    sens = lipschitz_estimate(w, shapes); wn = float(np.linalg.norm(w))
    viol = max(max(0.0, sens ** 2 - L ** 2), max(0.0, wn ** 2 - B ** 2))
    return dict(n_vars=n_params(shapes), iters=int(iters), time=float(dt),
                test=mse_numpy(w, Xte, yte, shapes), sensitivity=sens,
                wnorm=wn, violation=viol, success=bool(ok))


def main():
    rows = {}
    for H in SIZES:
        rows[H] = dict(H=H, ipopt=solve_ipopt(H), adam=run_adam(H))
        r = rows[H]
        print(f"H={H:>3} n={r['ipopt']['n_vars']:>3} | IPOPT t={r['ipopt']['time']:>7.2f}s "
              f"it={r['ipopt']['iters']:>4} test={r['ipopt']['test']:.4f} viol={r['ipopt']['violation']:.0e}"
              f" | Adam t={r['adam']['time']:>5.2f}s test={r['adam']['test']:.4f} sens={r['adam']['sensitivity']:.1f}")
    for H in SIZES_PENALTY:
        rows[H]["penalty"] = run_penalty(H)
        p = rows[H]["penalty"]
        print(f"   penalty-Adam H={H:>3}: t={p['time']:.2f}s it={p['iters']} "
              f"test={p['test']:.4f} sens={p['sensitivity']:.2f} viol={p['violation']:.0e}")

    out = dict(L=L, B=B, rho=RHO, sizes=SIZES, sizes_penalty=SIZES_PENALTY,
               rows=[rows[H] for H in SIZES])
    path = os.path.join(HERE, "results", "scaling_study.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
