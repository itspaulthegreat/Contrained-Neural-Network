# -*- coding: utf-8 -*-
"""Slide 6 figure: Case Study 1 (synthetic) — the true function, the noisy test
data, and ALL FOUR fits from results/fixed_spec_synth.json, so the picture
matches slide 5's table exactly. Fixed spec L=4, B=6, ordered biases, sigma=0.2.

    python synth_fits_figure.py  ->  figures/fig_synth_fits.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import param_shapes, forward_numpy
from src.data import make_teacher, _teacher_forward
import validation_protocol as vp

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "results", "fixed_spec_synth.json")))
rows = {r["method"].split(" (")[0]: r for r in d["rows"]}
shapes = param_shapes(1, vp.H, 1)

(Xtr, ytr), (Xva, yva), (Xte, yte) = vp.three_way_split(sigma=0.2)
teacher = make_teacher(seed=vp.SEED)
xs = np.linspace(-3, 3, 400).reshape(1, -1)

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(xs.ravel(), _teacher_forward(xs, *teacher).ravel(), "k--", lw=1.6, label="true function")
ax.plot(Xte.ravel(), yte.ravel(), "s", ms=4, color="0.6", label="test data (untouched)", zorder=1)

styles = [
    ("IPOPT", "tab:purple", "-",  2.6, "IPOPT (all 3, hard) — rate 4.0 ✓"),
    ("penalty-Adam", "tab:green", "-", 2.0, "penalty-Adam (soft) — rate 3.8 ✓"),
    ("AdamW", "tab:orange", "-.", 2.0, "AdamW (weight decay) — biases ✗"),
    ("plain Adam", "tab:red", ":", 2.0, "plain Adam — rate 9.6 ✗"),
]
for key, col, ls, lw, lab in styles:
    w = np.array(rows[key]["w"])
    ax.plot(xs.ravel(), forward_numpy(w, xs, shapes).ravel(), color=col, ls=ls, lw=lw, label=lab)

ax.set_xlabel("input  x")
ax.set_ylabel("output  y")
ax.set_ylim(-1.1, 1.1)
ax.set_title("Case Study 1 — the same fixed spec, four methods", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper center", ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(HERE, "figures", "fig_synth_fits.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print("wrote", out)
