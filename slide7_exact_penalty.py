# -*- coding: utf-8 -*-
"""Slide 7 (exact constraint vs soft penalty) at sigma=0.2, FIXED L=4 so the
bound binds (unconstrained reaches 9.57). Lipschitz-only, so the exact-penalty
threshold is a single clean lambda*.

Soft penalty:  min  MSE(w) + rho * max(0, ||w1||^2||w2||^2 - L^2)
Exact-penalty theory: the soft minimiser equals the constrained minimiser iff
rho > lambda* (the constraint's multiplier). We compute lambda* from IPOPT and
show the sweep crossing it.

    python slide7_exact_penalty.py  ->  figures/fig_slide7_exact_penalty.png
                                        results/slide7_exact_penalty.json
"""
import json
import os

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import (param_shapes, n_params, forward_symbolic,
                       unflatten_symbolic, random_init, mse_numpy)
from src.analysis import lipschitz_estimate
from src.penalty_adam import penalty_adam_optimize
import validation_protocol as vp

HERE = os.path.dirname(__file__)
L = 4.0
SIG = 0.2
H = vp.H
shapes = param_shapes(1, H, 1)
(Xtr, ytr), (Xva, yva), (Xte, yte) = vp.three_way_split(sigma=SIG)
w0 = random_init(shapes, scale=0.5, seed=vp.SEED)
te = lambda w: mse_numpy(w, Xte, yte, shapes)

# --- IPOPT, Lipschitz-only at L=4: get the multiplier lambda* ---
w = ca.MX.sym("w", n_params(shapes))
f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
g = ca.sumsqr(W1) * ca.sumsqr(W2)                       # ||W1||^2 ||W2||^2
s = ca.nlpsol("s", "ipopt", {"x": w, "f": f, "g": g},
              dict(ipopt=dict(max_iter=3000, tol=1e-10, print_level=0), print_time=False))
sol = s(x0=random_init(shapes, scale=0.5, seed=vp.SEED), lbg=-ca.inf, ubg=L ** 2)
w_ip = np.asarray(sol["x"]).flatten()
LAM = float(np.asarray(sol["lam_g"]).flatten()[0])
ip_rate = lipschitz_estimate(w_ip, shapes)
print(f"IPOPT (L=4): achieved rate {ip_rate:.4f}, lambda* = {LAM:.4e}, test {te(w_ip):.4f}")

# --- penalty-Adam sweep, Lipschitz-only ---
RHO = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
ach, tests = [], []
print("penalty-Adam sweep:")
for rho in RHO:
    out = penalty_adam_optimize(w0, shapes, Xtr, ytr, L_max=L, rho=float(rho),
                                lr=0.02, n_iter=3000, tol=0.0)
    a = lipschitz_estimate(out["w"], shapes)
    ach.append(a); tests.append(te(out["w"]))
    print(f"  rho={rho:<8g} rate={a:.4f}  test={te(out['w']):.4f}  "
          f"{'feasible' if a <= L + 1e-6 else 'VIOLATES'}")
ach = np.array(ach)
rho0 = lipschitz_estimate(penalty_adam_optimize(w0, shapes, Xtr, ytr, L_max=L, rho=0.0,
                          lr=0.02, n_iter=3000, tol=0.0)["w"], shapes)

json.dump(dict(L=L, sigma=SIG, lambda_star=LAM, ip_rate=ip_rate, ip_test=te(w_ip),
               rho=RHO.tolist(), achieved=ach.tolist(), tests=tests, rho0_rate=rho0),
          open(os.path.join(HERE, "results", "slide7_exact_penalty.json"), "w"), indent=1)

# --- figure ---
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(8.8, 4.9))
ax.axhspan(L, ach.max() * 1.15, color="tab:red", alpha=0.06)
ax.axhline(L, color="black", ls="--", lw=2)
ax.text(RHO[-1], L * 1.03, "bound L = 4", ha="right", fontsize=10, fontweight="bold")
ax.axvline(LAM, color="tab:purple", lw=2)
ax.text(LAM * 1.3, ach.max() * 0.9, f"λ* = {LAM:.1e}\n(the multiplier\nIPOPT returns)",
        color="tab:purple", fontsize=9.5, fontweight="bold")
ax.plot(RHO, ach, "o-", color="tab:olive", lw=2.2, ms=8,
        label="penalty-Adam: achieved sensitivity")
ax.plot([1.0], [ip_rate], "*", color="tab:purple", ms=20, zorder=5,
        label=f"IPOPT, hard constraint: {ip_rate:.2f} — no ρ to choose")
ax.set_xscale("log")
ax.set_xlabel("penalty weight  ρ   (log scale)")
ax.set_ylabel("achieved sensitivity  ‖w₁‖‖w₂‖")
ax.set_title("Soft penalty vs hard constraint (same bound L = 4): the penalty matches\n"
             "the constraint only once ρ passes λ* — which IPOPT returns for free",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig_slide7_exact_penalty.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print("wrote", out)
