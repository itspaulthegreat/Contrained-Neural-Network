"""
Depth study: does a DEEPER network change anything?

The professor's "enlarge the network" also means depth. On the synthetic task I
grow the number of hidden layers (width 8 each) from 1 to 3, keeping the same
sensitivity budget (Lipschitz product bound) and norm ball, and solve with IPOPT.
A plain-Adam baseline (run to convergence) is included for reference. The
question is whether depth buys accuracy, and what it costs the constrained solve.

    python -m experiments.depth_study   ->  results/depth_study.json
"""
import json
import os
import sys
import time

import numpy as np
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_dataset
from src.deep_model import (deep_shapes, n_params, forward_symbolic, mse_numpy,
                            random_init, lipschitz_product_symbolic, lipschitz_estimate)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, B = 4.0, 6.0
WIDTH = 8
DEPTHS = [1, 2, 3]                       # number of hidden layers
TOL, CAP = 1e-9, 40000
IPOPT_OPTS = dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0), print_time=False)

Xtr, ytr, Xte, yte = generate_dataset(n_train=60, n_test=40, noise_std=0.05, seed=0)


def solve_ipopt(shapes):
    w = ca.MX.sym("w", n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    g = ca.vertcat(lipschitz_product_symbolic(w, shapes), ca.sumsqr(w))
    solver = ca.nlpsol("deep", "ipopt", {"x": w, "f": f, "g": g}, IPOPT_OPTS)
    t0 = time.time()
    sol = solver(x0=random_init(shapes, seed=0),
                 lbg=[-ca.inf, -ca.inf], ubg=[L ** 2, B ** 2])
    dt = time.time() - t0
    return np.asarray(sol["x"]).flatten(), int(solver.stats()["iter_count"]), dt, \
        bool(solver.stats()["success"])


def adam_mse(shapes, lr=0.02):
    """Unconstrained Adam on the MSE, run to a loss plateau or the cap."""
    w = ca.MX.sym("w", n_params(shapes))
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    f_fn = ca.Function("f", [w], [f])
    g_fn = ca.Function("g", [w], [ca.gradient(f, w)])
    wv = random_init(shapes, seed=0)
    m = np.zeros_like(wv); v = np.zeros_like(wv); prev = None
    t0 = time.time(); used = CAP
    for t in range(1, CAP + 1):
        grad = np.asarray(g_fn(wv)).flatten()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        wv = wv - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        loss = float(f_fn(wv))
        if prev is not None and abs(prev - loss) < TOL:
            used = t; break
        prev = loss
    return wv, used, time.time() - t0


def main():
    rows = []
    for depth in DEPTHS:
        shapes = deep_shapes(1, [WIDTH] * depth, 1)
        n = n_params(shapes)

        w_ip, it_ip, t_ip, ok = solve_ipopt(shapes)
        sens_ip = lipschitz_estimate(w_ip, shapes)
        wn_ip = float(np.linalg.norm(w_ip))
        viol = max(max(0.0, sens_ip ** 2 - L ** 2), max(0.0, wn_ip ** 2 - B ** 2))

        w_ad, it_ad, t_ad = adam_mse(shapes)
        sens_ad = lipschitz_estimate(w_ad, shapes)

        row = dict(
            depth=depth, n_params=n,
            ipopt=dict(test=mse_numpy(w_ip, Xte, yte, shapes),
                       train=mse_numpy(w_ip, Xtr, ytr, shapes),
                       iters=it_ip, time=t_ip, sensitivity=sens_ip,
                       wnorm=wn_ip, violation=viol, success=ok),
            adam=dict(test=mse_numpy(w_ad, Xte, yte, shapes),
                      train=mse_numpy(w_ad, Xtr, ytr, shapes),
                      iters=it_ad, time=t_ad, sensitivity=sens_ad),
        )
        rows.append(row)
        print(f"depth={depth} layers  n={n:>3}  "
              f"IPOPT test={row['ipopt']['test']:.4f} sens={sens_ip:.2f} "
              f"it={it_ip} t={t_ip:.2f}s viol={viol:.1e} ok={ok}  |  "
              f"Adam test={row['adam']['test']:.4f} sens={sens_ad:.1f} it={it_ad}")

    out = dict(L=L, B=B, width=WIDTH, depths=DEPTHS, rows=rows)
    path = os.path.join(HERE, "results", "depth_study.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
