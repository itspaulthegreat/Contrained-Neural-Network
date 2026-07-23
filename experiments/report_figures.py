"""
Regenerate the figures used in the written report from the stored result files
in ``results/``. Run the corresponding studies first, then

    python -m experiments.report_figures

writes fig_exact_penalty.png, fig_scaling.png and fig_pendulum.png into
``figures/``.
"""
import glob
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


def _size_rows(pattern, keys):
    rows = []
    for f in glob.glob(os.path.join(RESULTS, pattern)):
        d = json.load(open(f))
        rows.append(tuple(d[k] for k in keys))
    return sorted(rows)


def fig_scaling():
    # (H, n, time, test) for constrained IPOPT and unconstrained Adam
    ip = _size_rows("exp_size_H*.json", ["H", "n_vars", "solve_time", "test_mse"])
    ad = _size_rows("exp_complexity_adam_H*.json", ["H", "n_vars", "solve_time", "test_mse"])
    Hi = [r[0] for r in ip]; ti = [r[2] for r in ip]; tei = [r[3] for r in ip]
    Ha = [r[0] for r in ad]; ta = [r[2] for r in ad]

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.plot(Hi, ti, "o-", color="tab:purple", lw=2, label="IPOPT (constrained), solve time")
    ax.plot(Ha, ta, "s-", color="tab:green", lw=2, label="Adam (unconstrained), solve time")
    ax.set_yscale("log")
    ax.set_xlabel("hidden units $H$  (parameters $n=3H+1$)")
    ax.set_ylabel("solve time [s]")
    ax.legend(fontsize=8, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(Hi, tei, "^--", color="tab:gray", lw=1.5, alpha=0.8, label="IPOPT test MSE")
    ax2.set_ylabel("test MSE [-]")
    ax2.set_ylim(0, max(tei) * 3)
    ax2.grid(False)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_scaling.png")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_pendulum():
    from experiments.pendulum_protocol import three_way_split_pendulum, H
    Z = np.load(os.path.join(RESULTS, "pendulum_traj.npz"))
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), _, _ = three_way_split_pendulum()
    ts = np.linspace(0, 6, 400).reshape(1, -1)
    true = pendulum_true(ts).ravel()

    methods = [("IPOPT (hard, all 3)", "ip", "tab:purple", "-"),
               ("penalty-Adam", "pen", "tab:olive", "--"),
               ("AdamW", "aw", "tab:orange", "-."),
               ("plain Adam", "pl", "tab:red", ":")]
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
    fig_pendulum()


if __name__ == "__main__":
    main()
