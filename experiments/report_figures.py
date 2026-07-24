"""
Regenerate the figures used in the written report from the stored result files
in ``results/``. Run the corresponding studies first, then

    python -m experiments.report_figures

writes fig_exact_penalty.png, fig_scaling.png and fig_pendulum.png into
``figures/``.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import param_shapes, forward_numpy
from src.data import pendulum_true

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load(name):
    return json.load(open(os.path.join(RESULTS, name), encoding="utf-8"))


def fig_exact_penalty():
    d = _load("exact_penalty.json")
    rho = np.array(d["rho"]); ach = np.array(d["achieved"])
    L, lam, ip_rate = d["L"], d["lambda_star"], d["ip_rate"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.axhline(L, color="black", ls="--", lw=1.8, label=f"bound $L={L:g}$")
    ax.axvline(lam, color="tab:purple", lw=1.5)
    ax.plot(rho, ach, "o-", color="tab:olive", lw=2, ms=6,
            label="penalty solution (solved to convergence)")
    ax.plot(rho[-1], ip_rate, "*", color="tab:purple", ms=16, zorder=5,
            label=f"hard constraint: {ip_rate:.2f}")
    ax.annotate(r"$\lambda^\star$", xy=(lam, L), xytext=(lam * 1.6, L + 1.6),
                color="tab:purple", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="tab:purple"))
    ax.set_xscale("log")
    ax.set_xlabel(r"penalty weight $\rho$ [-]")
    ax.set_ylabel(r"achieved sensitivity $\|w_1\|\,\|w_2\|$ [-]")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_exact_penalty.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


_SCALE_SERIES = [("ipopt", "IPOPT (hard)", "tab:purple", "o"),
                 ("penalty", "penalty-Adam (soft, same reqs)", "tab:olive", "D")]


def fig_scaling():
    """Matched width scaling: the same Lipschitz+ball problem, three methods."""
    rows = _load("scaling_study.json")["rows"]

    def series(method, key):
        xs, ys = [], []
        for r in rows:
            if method in r:
                xs.append(r["H"]); ys.append(r[method][key])
        return xs, ys

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.7))
    for m, lab, col, mk in _SCALE_SERIES:
        xs, ys = series(m, "time")
        a.plot(xs, ys, mk + "-", color=col, lw=2, ms=5, label=lab)
    a.set_yscale("log"); a.set_xlabel("hidden units $H$  ($n=3H+1$)")
    a.set_ylabel("solve time [s]"); a.set_title("Cost"); a.legend(fontsize=7.5, loc="upper left")

    mx = 0
    for m, lab, col, mk in _SCALE_SERIES:
        xs, ys = series(m, "test")
        b.plot(xs, ys, mk + "-", color=col, lw=2, ms=5, label=lab)
        mx = max(mx, max(ys) if ys else 0)
    b.set_xlabel("hidden units $H$  ($n=3H+1$)"); b.set_ylabel("test MSE [-]")
    b.set_ylim(0, mx * 1.25); b.set_title("Accuracy"); b.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_scaling.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_depth():
    """Sensitivity vs depth: the constraint pins it, unconstrained Adam explodes."""
    d = _load("depth_study.json")
    rows = d["rows"]; L = d["L"]
    depth = [r["depth"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.axhline(L, color="black", ls="--", lw=1.5, label=f"bound $L={L:g}$")
    ax.plot(depth, [r["ipopt"]["sensitivity"] for r in rows], "o-", color="tab:purple",
            lw=2, ms=7, label="IPOPT (constrained)")
    ax.plot(depth, [r["adam"]["sensitivity"] for r in rows], "s-", color="tab:red",
            lw=2, ms=7, label="plain Adam (unconstrained)")
    ax.set_yscale("log"); ax.set_xticks(depth)
    ax.set_xlabel("number of hidden layers (depth)")
    ax.set_ylabel(r"achieved sensitivity $\prod_i \|W_i\|_F$ [-]")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_depth.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_pendulum():
    from experiments.pendulum_protocol import three_way_split_pendulum, H
    Z = np.load(os.path.join(RESULTS, "pendulum_traj.npz"))
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), _, _ = three_way_split_pendulum()
    ts = np.linspace(0, 6, 400).reshape(1, -1)
    true = pendulum_true(ts).ravel()

    methods = [("IPOPT (hard, Lip+ball)", "ip", "tab:purple", "-"),
               ("penalty-Adam (soft, same reqs)", "pen", "tab:olive", "--")]
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.scatter(Xtr.ravel(), ytr.ravel(), s=16, color="0.6", label="noisy data", zorder=1)
    ax.plot(ts.ravel(), true, "k--", lw=1.6, label="true response", zorder=2)
    for name, key, col, ls in methods:
        w = Z[key][-1]
        fit = forward_numpy(w, ts, shapes).ravel()
        ax.plot(ts.ravel(), fit, ls, color=col, lw=2, label=name, zorder=3)
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel(r"angle $\theta$ [rad]")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_pendulum.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def main():
    fig_exact_penalty()
    fig_scaling()
    fig_depth()
    fig_pendulum()


if __name__ == "__main__":
    main()
