# -*- coding: utf-8 -*-
"""A small figure for slide 3: the three baselines written as proper equations,
building from plain training to the soft-penalty version, then pointing at the
hard-constraint alternative. Typeset with matplotlib mathtext.

    python baseline_equations_figure.py  ->  figures/fig_baseline_equations.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")

plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 13})

fig = plt.figure(figsize=(11.6, 4.7))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

NAVY = "#1f3b73"; TEAL = "#11727c"; GRAY = "#606060"; GREEN = "#1e7d32"

rows = [
    # (badge, label, equation, note, color)
    ("1", "Plain Adam — just fit the data",
     r"$\min_{w}\ \ f(w)=\dfrac{1}{N}\sum_{i=1}^{N}(\hat{y}(x_i;w)-y_i)^2$",
     "no constraint, no regularizer — sensitivity is whatever this happens to produce", GRAY),
    ("2", "AdamW — add weight decay (the standard fix)",
     r"$\min_{w}\ \ f(w)\ +\ \alpha\,\Vert w\Vert_2^{2}$",
     r"one knob $\alpha$; shrinks all weights, but cannot set any property to a target", GRAY),
    ("3", "Penalty-Adam — all three requirements as soft hinges",
     r"$\min_{w}\ f(w)+\rho\,[\,\max(0,\Vert W_1\Vert_F^{2}\Vert W_2\Vert_F^{2}-L^2)"
     r"+\max(0,\Vert w\Vert_2^{2}-B^2)+\sum_j \max(0,\,b_{1,j}-b_{1,j+1})\,]$",
     r"one shared $\rho$; zero inside each bound, pushes back only once it is exceeded", TEAL),
]

y = 0.90
for badge, label, eqn, note, col in rows:
    ax.add_patch(plt.Circle((0.035, y - 0.02), 0.022, transform=ax.transAxes,
                            color=TEAL, zorder=3))
    ax.text(0.035, y - 0.02, badge, transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=12, fontweight="bold", zorder=4)
    ax.text(0.075, y, label, transform=ax.transAxes, ha="left", va="center",
            color=NAVY, fontsize=12.5, fontweight="bold")
    ax.text(0.075, y - 0.075, eqn, transform=ax.transAxes, ha="left", va="center",
            color="black", fontsize=13.5 if badge != "3" else 11.6)
    ax.text(0.075, y - 0.145, note, transform=ax.transAxes, ha="left", va="center",
            color=col, fontsize=10, style="italic")
    y -= 0.30

ax.text(0.075, 0.03,
        r"This project: make each bracket a HARD constraint instead of a penalty — "
        r"then it holds exactly, with no $\rho$ to choose.",
        transform=ax.transAxes, ha="left", va="center", color=GREEN,
        fontsize=11.5, fontweight="bold")

out = os.path.join(FIG, "fig_baseline_equations.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", out)
