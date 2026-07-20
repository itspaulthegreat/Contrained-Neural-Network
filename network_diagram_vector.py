# -*- coding: utf-8 -*-
"""Vector-notation network diagram (Léo: single input x -> use vectors w1,w2 in
R^8, no Frobenius/operator norm). Shared by slide 2 (formulation) and slide 4
(derivation).

    x --w1--> (+b1) --tanh--> --w2--> (+b2) --> yhat
    yhat(x) = w2^T tanh(w1 x + b1) + b2,   w1,w2 in R^8

    python network_diagram_vector.py  ->  figures/fig_network_vector.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(__file__)
GREEN, BLUE, GREY, INK = "#2e8b57", "#2f6fb0", "#e8ebef", "#1f3b73"

fig, ax = plt.subplots(figsize=(9.2, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")


def box(x, w, label, color, txtcolor="white", fs=15):
    ax.add_patch(FancyBboxPatch((x, 1.2), w, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none"))
    ax.text(x + w / 2, 1.65, label, ha="center", va="center", color=txtcolor,
            fontsize=fs, fontweight="bold")


def arrow(x0, x1):
    ax.add_patch(FancyArrowPatch((x0, 1.65), (x1, 1.65), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color="0.35"))


ax.text(0.35, 1.65, "x", ha="center", va="center", fontsize=16, style="italic", color=INK)
arrow(0.6, 1.5)
box(1.5, 1.7, "w₁ x + b₁", GREEN);
arrow(3.2, 4.1)
box(4.1, 1.5, "tanh", GREY, txtcolor="#333", fs=14)
arrow(5.6, 6.5)
box(6.5, 1.7, "w₂ᵀ(·) + b₂", BLUE)
arrow(8.2, 9.1)
ax.text(9.35, 1.65, "ŷ", ha="center", va="center", fontsize=16, style="italic", color=INK)

# norm annotations under the affine boxes
ax.text(2.35, 0.75, "‖w₁‖₂", ha="center", fontsize=13, color=GREEN, fontweight="bold")
ax.text(4.85, 0.75, "slope ≤ 1", ha="center", fontsize=11, color="#555")
ax.text(7.35, 0.75, "‖w₂‖₂", ha="center", fontsize=13, color=BLUE, fontweight="bold")

ax.text(5.0, 2.75, "ŷ(x) = w₂ᵀ tanh(w₁x + b₁) + b₂     with   w₁, w₂ ∈ ℝ⁸",
        ha="center", fontsize=14, color=INK, fontweight="bold")
ax.text(5.0, 0.25, "sensitivity  |ŷ′(x)| ≤ ‖w₁‖₂ · ‖w₂‖₂   (product of two vector norms — no Frobenius needed)",
        ha="center", fontsize=11, color="#444")

fig.tight_layout()
out = os.path.join(HERE, "figures", "fig_network_vector.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
