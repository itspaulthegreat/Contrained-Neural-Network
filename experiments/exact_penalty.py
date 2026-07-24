# -*- coding: utf-8 -*-
"""Exact hinge-penalty experiment for the presentation.

The hinge problem

    min_w MSE(w) + rho * max(0, ||W1||_F^2 ||W2||_F^2 - L^2)

is written with an epigraph variable s:

    min_{w,s} MSE(w) + rho*s
    subject to s >= 0,
               s >= ||W1||_F^2 ||W2||_F^2 - L^2.

This is exactly the same nonsmooth hinge objective after minimizing over s,
but it lets IPOPT solve every sweep point to convergence.  Using the same
solver for the hard and penalty formulations isolates the formulation effect
instead of mixing it with a fixed-budget Adam endpoint.

    python exact_penalty.py
        -> figures/fig_exact_penalty.png
        -> results/exact_penalty.json
"""
import json
import os

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis import lipschitz_estimate
from src.model import (forward_symbolic, mse_numpy, n_params, param_shapes,
                       random_init, unflatten_symbolic)
from experiments import synthetic_protocol as vp


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 4.0
SIG = 0.2
H = vp.H
IPOPT_OPTIONS = dict(
    ipopt=dict(max_iter=3000, tol=1e-10, acceptable_tol=1e-9, print_level=0),
    print_time=False,
)

shapes = param_shapes(1, H, 1)
(Xtr, ytr), (Xva, yva), (Xte, yte) = vp.three_way_split(sigma=SIG)
w0 = random_init(shapes, scale=0.5, seed=vp.SEED)
test_mse = lambda weights: mse_numpy(weights, Xte, yte, shapes)
train_mse = lambda weights: mse_numpy(weights, Xtr, ytr, shapes)

w = ca.MX.sym("w", n_params(shapes))
f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
violation = ca.sumsqr(W1) * ca.sumsqr(W2) - L ** 2
violation_fn = ca.Function("violation", [w], [violation])


# Hard-constrained reference and its multiplier.
hard_solver = ca.nlpsol(
    "hard", "ipopt", {"x": w, "f": f, "g": violation}, IPOPT_OPTIONS
)
hard_sol = hard_solver(x0=w0, lbg=-ca.inf, ubg=0.0)
w_ip = np.asarray(hard_sol["x"]).flatten()
lambda_star = float(np.asarray(hard_sol["lam_g"]).flatten()[0])
ip_rate = lipschitz_estimate(w_ip, shapes)


def solve_exact_hinge(rho):
    """Solve MSE + rho*max(0, violation) via its exact epigraph NLP."""
    slack = ca.MX.sym("slack")
    z = ca.vertcat(w, slack)
    constraints = ca.vertcat(slack, slack - violation)
    solver_name = (
        "pen_"
        + f"{rho:.3e}".replace(".", "p").replace("-", "m").replace("+", "p")
    )
    solver = ca.nlpsol(
        solver_name,
        "ipopt",
        {"x": z, "f": f + float(rho) * slack, "g": constraints},
        IPOPT_OPTIONS,
    )
    initial_violation = float(violation_fn(w0))
    sol = solver(
        x0=np.r_[w0, max(0.0, initial_violation)],
        lbg=np.zeros(2),
        ubg=np.full(2, np.inf),
    )
    z_opt = np.asarray(sol["x"]).flatten()
    weights = z_opt[:-1]
    return weights, float(z_opt[-1]), solver.stats()["return_status"]


# Include several points around lambda* so the threshold is visible.
RHO = np.array([
    5e-5, 8e-5, 1e-4, 1.2e-4, 1.4e-4, 1.5e-4,
    1.7e-4, 2e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0,
])
achieved, train_errors, test_errors, slacks, statuses = [], [], [], [], []

print(
    f"Hard IPOPT: rate={ip_rate:.9f}, lambda*={lambda_star:.9e}, "
    f"train={train_mse(w_ip):.9f}, test={test_mse(w_ip):.9f}"
)
print("Exact hinge sweep (epigraph/slack formulation):")
for rho in RHO:
    weights, slack, status = solve_exact_hinge(float(rho))
    rate = lipschitz_estimate(weights, shapes)
    achieved.append(rate)
    train_errors.append(train_mse(weights))
    test_errors.append(test_mse(weights))
    slacks.append(slack)
    statuses.append(status)
    print(
        f"  rho={rho:<9g} rate={rate:.9f} slack={slack:.3e} "
        f"train={train_errors[-1]:.9f} status={status}"
    )

achieved = np.asarray(achieved)
result = dict(
    L=L,
    sigma=SIG,
    solver="IPOPT epigraph solve of the exact hinge penalty",
    lambda_star=lambda_star,
    ip_rate=ip_rate,
    ip_train=train_mse(w_ip),
    ip_test=test_mse(w_ip),
    rho=RHO.tolist(),
    achieved=achieved.tolist(),
    train=train_errors,
    tests=test_errors,
    slacks=slacks,
    statuses=statuses,
)
with open(os.path.join(HERE, "results", "exact_penalty.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=1)


plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(8.8, 4.9))
ax.axhline(L, color="black", ls="--", lw=2, label="bound L = 4")
ax.axvline(lambda_star, color="tab:purple", lw=2)
ax.plot(
    RHO,
    achieved,
    "o-",
    color="tab:olive",
    lw=2.2,
    ms=7,
    label="converged exact-hinge solution",
)
ax.plot(
    [RHO[-1]],
    [ip_rate],
    "*",
    color="tab:purple",
    ms=20,
    zorder=5,
    label=f"hard-constrained IPOPT reference: {ip_rate:.2f}",
)
ax.annotate(
    f"lambda* = {lambda_star:.1e}",
    xy=(lambda_star, L),
    xytext=(8, 34),
    textcoords="offset points",
    color="tab:purple",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="tab:purple"),
)
ax.text(
    2.2e-4,
    L * 1.015,
    "rho > lambda*: penalty optimum stays on the boundary",
    fontsize=9.5,
    fontweight="bold",
)
ax.set_xscale("log")
ax.set_xlabel("penalty weight rho (log scale)")
ax.set_ylabel("achieved sensitivity  ||W1||F ||W2||F")
ax.set_title(
    "Exact hinge penalty vs hard constraint (same bound L = 4)\n"
    "Once rho passes lambda*, increasing rho does not push the solution below L",
    fontsize=11.5,
    fontweight="bold",
)
ax.legend(fontsize=8.8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()

figure_path = os.path.join(HERE, "figures", "fig_exact_penalty.png")
fig.savefig(figure_path, dpi=200)
plt.close(fig)
print("wrote", figure_path)
print("wrote", os.path.join(HERE, "results", "exact_penalty.json"))
