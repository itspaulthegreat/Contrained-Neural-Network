# -*- coding: utf-8 -*-
"""Slide 8 figure: the simulated pendulum data, the true response, and ALL FOUR
fits from results/fixed_spec_pendulum.json (IPOPT all-three, penalty-Adam,
AdamW, plain Adam) so the graph matches the table exactly. Fixed L = 4, B = 6.

    python pendulum_L4_figure.py  ->  figures/fig_pendulum_L4.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import param_shapes, forward_numpy
from src.data import pendulum_true
import pendulum_protocol as pp

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "results", "fixed_spec_pendulum.json")))
rows = {r["method"].split(" (")[0]: r for r in d["rows"]}
shapes = param_shapes(1, pp.H, 1)
tt = np.linspace(0, 6, 400).reshape(1, -1)

# the data (same split the comparison used)
(Xtr, ytr), _, _ = pp.three_way_split_pendulum()

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(tt.ravel(), pendulum_true(tt).ravel(), "k--", lw=1.6, label="true response θ(t)")
ax.plot(Xtr.ravel(), ytr.ravel(), "o", ms=4, color="0.6", label="noisy measurements", zorder=1)

styles = [
    ("IPOPT", "tab:purple", "-",  2.6, "IPOPT (all 3, hard) — rate 4.0 ✓"),
    ("penalty-Adam", "tab:green", "-", 2.0, "penalty-Adam (soft) — rate 3.8 ✓"),
    ("AdamW", "tab:orange", "-.", 2.0, "AdamW (weight decay) — rate 7.3 ✗"),
    ("plain Adam", "tab:red", ":", 2.0, "plain Adam — rate 8.3 ✗"),
]
for key, col, ls, lw, lab in styles:
    w = np.array(rows[key]["w"])
    ax.plot(tt.ravel(), forward_numpy(w, tt, shapes).ravel(), color=col, ls=ls, lw=lw, label=lab)

ax.set_xlabel("time  t  [s]")
ax.set_ylabel("angle  θ  [rad]")
ax.set_title("Fixed spec L = 4 rad/s, ‖w‖ ≤ 6, ordered biases", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right", ncol=1)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig_pendulum_L4.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print("wrote", out)
