# -*- coding: utf-8 -*-
"""Slide 3 (The Problem) figure, simplified per Léo/ChatGPT: Adam ONLY (no
Gauss-Newton, no H sweeps). Standard training's sensitivity drifts with the
noise and walks straight through the bound a task might require.

    python problem_figure_simple.py  ->  figures/fig_problem_simple.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
load = lambda n: json.load(open(os.path.join(HERE, "results", n + ".json")))

sig = [0.0, 0.1, 0.2, 0.3]
vals = [load(f"exp_noise_{t}_adam")["lipschitz_estimate"]
        for t in ["0p0", "0p1", "0p2", "0p3"]]

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(7.6, 4.7))
x = np.arange(len(sig))
colors = ["tab:green" if v <= 4 else "tab:red" for v in vals]
ax.bar(x, vals, color=colors, width=0.6)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.3, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(4, color="black", ls="--", lw=2)
ax.text(len(sig) - 0.5, 4.4, "a bound a task might require (L = 4)",
        ha="right", fontsize=10.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"σ = {s}" for s in sig])
ax.set_xlabel("label noise")
ax.set_ylabel("sensitivity  ‖w₁‖ · ‖w₂‖")
ax.set_title("Standard training (Adam): the sensitivity is whatever the data makes it",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig_problem_simple.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print("wrote", out)
