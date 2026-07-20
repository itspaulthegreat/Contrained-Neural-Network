# -*- coding: utf-8 -*-
"""The SAME fixed spec on both case studies (2026-07-20 restructuring):
    Lipschitz  ||w1||*||w2|| <= L = 4      (rate, in rad/s for the pendulum)
    Norm ball  ||w||_2       <= B = 6
    Symmetry   b1 ordered
No bound is validation-selected -- L and B are fixed up front. The Adam-based
baselines still sweep their own knob (rho, wd); we report how many swept
settings complied with ALL three constraints.

Writes results/fixed_spec_synth.json and results/fixed_spec_pendulum.json, each
with one representative row per algorithm (IPOPT, penalty-Adam, AdamW, plain
Adam) plus compliance counts for the swept methods.

    python fixed_spec_comparison.py
"""
import json
import os
import time

import numpy as np

from src.model import param_shapes, mse_numpy, unflatten_numpy, random_init
from src.analysis import lipschitz_estimate
from src.penalty_adam import multi_penalty_adam_optimize
from src.baseline_adam import adam_optimize

L, B = 4.0, 6.0
RHO_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
WD_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_ITER = 3000
TOL = 1e-6
HERE = os.path.dirname(__file__)


def checks(w, shapes):
    _, b1, _, _ = unflatten_numpy(w, shapes)
    rate = lipschitz_estimate(w, shapes)
    wn = float(np.linalg.norm(w))
    sym = float(np.diff(b1.flatten()).min())
    return dict(rate=rate, wnorm=wn, sym=sym,
                lip_ok=bool(rate <= L + TOL), ball_ok=bool(wn <= B + TOL),
                sym_ok=bool(sym >= -TOL),
                all_ok=bool(rate <= L + TOL and wn <= B + TOL and sym >= -TOL))


def run_case(name, split, ipopt_fn, H):
    """split -> ((Xtr,ytr),(Xva,yva),(Xte,yte)); ipopt_fn(Xtr,ytr) -> (w,iters,time)."""
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split
    w0 = random_init(shapes, scale=0.5, seed=2 if name == "synth" else 0)
    te = lambda w: mse_numpy(w, Xte, yte, shapes)
    va = lambda w: mse_numpy(w, Xva, yva, shapes)

    def row(method, w, iters, tsec, extra=""):
        c = checks(w, shapes)
        c.update(method=method, test=te(w), iters=iters, time=tsec, note=extra,
                 w=np.asarray(w).flatten().tolist())
        return c

    rows = []

    # --- IPOPT: all three hard, fixed L=4, B=6 ---
    w_ip, it_ip, t_ip = ipopt_fn(Xtr, ytr)
    rows.append(row("IPOPT (all three, hard)", w_ip, it_ip, t_ip, "by construction"))

    # --- penalty-Adam: all three soft, sweep rho, count compliance ---
    n_pen = 0; best_pen = None
    for rho in RHO_GRID:
        out = multi_penalty_adam_optimize(w0, shapes, Xtr, ytr, rho=rho, L_max=L,
                                          B_max=B, symmetry=True, n_iter=N_ITER)
        c = checks(out["w"], shapes)
        n_pen += c["all_ok"]
        if c["all_ok"] and (best_pen is None or va(out["w"]) < best_pen[0]):
            best_pen = (va(out["w"]), out["w"])
    w_pen = best_pen[1] if best_pen else out["w"]
    rows.append(row("penalty-Adam (all three, soft)", w_pen, N_ITER, None,
                    f"{n_pen}/{len(RHO_GRID)} ρ complied"))

    # --- AdamW: sweep wd, count compliance ---
    n_aw = 0; best_aw = None
    for wd in WD_GRID:
        w = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=N_ITER,
                          weight_decay=wd, tol=0.0)["w"]
        c = checks(w, shapes)
        n_aw += c["all_ok"]
        if best_aw is None or va(w) < best_aw[0]:
            best_aw = (va(w), w)
    rows.append(row("AdamW (weight decay)", best_aw[1], N_ITER, None,
                    f"{n_aw}/{len(WD_GRID)} wd complied"))

    # --- plain Adam ---
    w_pl = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=N_ITER,
                         weight_decay=0.0, tol=0.0)["w"]
    rows.append(row("plain Adam (nothing)", w_pl, N_ITER, None,
                    "1/1" if checks(w_pl, shapes)["all_ok"] else "0/1"))

    out = dict(case=name, L=L, B=B, H=H, rho_grid=RHO_GRID, wd_grid=WD_GRID, rows=rows)
    path = os.path.join(HERE, "results", f"fixed_spec_{name}.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n=== {name}  (L={L}, B={B}, all three) ===")
    for r in rows:
        m = lambda ok: "✓" if ok else "✗"
        print(f"  {r['method']:<32} rate={r['rate']:>8.3f}{m(r['lip_ok'])}  "
              f"|w|={r['wnorm']:>7.3f}{m(r['ball_ok'])}  sym={r['sym']:>+8.4f}{m(r['sym_ok'])}  "
              f"test={r['test']:.4f}  [{r['note']}]")
    print("wrote", path)
    return out


if __name__ == "__main__":
    import validation_protocol as vp
    import pendulum_protocol as pp

    # synthetic, sigma=0.2
    def synth_ipopt(Xtr, ytr):
        t0 = time.time(); w = vp.fit_general(Xtr, ytr, L=L, B=B, sym=True); return w, None, time.time() - t0
    run_case("synth", vp.three_way_split(sigma=0.2), synth_ipopt, vp.H)

    # pendulum, sigma=0.15, all three via the patched fit_ipopt (B passed)
    def pend_ipopt(Xtr, ytr):
        return pp.fit_ipopt(Xtr, ytr, L, B=B, full=True)
    run_case("pendulum", pp.three_way_split_pendulum(), pend_ipopt, pp.H)
