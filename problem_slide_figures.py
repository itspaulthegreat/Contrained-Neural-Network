"""
problem_slide_figures.py
──────────────────────────
Two images for the "optimization problem" slide, so the maths is TYPESET
(real fractions, real norms, real subscripts) instead of typed as plain
text, and so the network is something the room can SEE:

  figures/fig_nlp_statement.png  — the full NLP, mathtext-rendered
  figures/fig_network.png        — the architecture with every parameter
                                   block labelled and COUNTED

Parameter counts are computed from src.model.param_shapes, never typed by
hand: for H = 8 they are W1 (8x1) + b1 (8) = 16 and W2 (1x8) + b2 (1) = 9,
so w has 25 entries -- the decision vector of the NLP.

    python problem_slide_figures.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

from src.model import param_shapes, n_params

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, 'figures')
H = 8

NAVY = '#1f3b73'
PURPLE = '#6a1b9a'
GREEN = '#1e7d32'


def _counts():
    """(n_layer1, n_layer2, n_total) derived from the real shape helper:
    W1 is (H x d_in) with bias b1 (H); W2 is (d_out x H) with bias b2 (d_out)."""
    shapes = param_shapes(1, H, 1)
    d_in, h, d_out = shapes['d_in'], shapes['H'], shapes['d_out']
    n1 = h * d_in + h
    n2 = d_out * h + d_out
    total = n_params(shapes)
    assert n1 + n2 == total, (n1, n2, total)
    return n1, n2, total


def nlp_statement():
    n1, n2, n_tot = _counts()
    fig = plt.figure(figsize=(8.0, 5.4))
    fig.patch.set_facecolor('white')
    X_LAB, X_MATH = 0.05, 0.29

    fig.text(X_LAB, 0.945, 'minimize', va='center', fontsize=15, color=NAVY,
             fontweight='bold')
    fig.text(X_MATH, 0.945, r'$f(w)\;=\;\dfrac{1}{N}\,\sum_{i=1}^{N}\,'
                            r'\left(\hat{y}(x_i;\,w)\;-\;y_i\right)^{2}$',
             va='center', fontsize=19)
    fig.text(X_LAB + 0.045, 0.855, r'$w\,\in\,\mathbb{R}^{%d}$' % n_tot,
             va='center', fontsize=13, color=PURPLE)
    fig.text(X_MATH, 0.775, 'the mean squared error on the training points',
             va='center', fontsize=10.5, color='dimgray')

    fig.text(X_LAB, 0.665, 'subject to', va='center', fontsize=15, color=NAVY,
             fontweight='bold')
    rows = [
        (0.665, r'$\Vert W_1\Vert_F^{2}\;\cdot\;\Vert W_2\Vert_F^{2}\;\leq\;L_{max}^{2}$',
         'Lipschitz bound — bilinear, NONCONVEX'),
        (0.480, r'$\Vert w\Vert_2^{2}\;\leq\;B_{max}^{2}$',
         'norm ball — convex quadratic'),
        (0.310, r'$b_{1,j}\;-\;b_{1,j+1}\;\leq\;0,\qquad j=1\,\ldots\,H-1$',
         'symmetry breaking — linear'),
    ]
    for y, formula, note in rows:
        fig.text(X_MATH, y, formula, va='center', fontsize=17, color=PURPLE)
        fig.text(X_MATH, y - 0.072, note, va='center', fontsize=10, color='dimgray')

    fig.text(X_LAB, 0.150, 'where', va='center', fontsize=14, color=NAVY,
             fontweight='bold')
    fig.text(X_MATH, 0.150, r'$\hat{y}(x;w)\;=\;W_2\,\tanh(W_1x+b_1)\;+\;b_2$',
             va='center', fontsize=16)
    fig.text(X_MATH, 0.045, r'$w=(W_1,\,b_1,\,W_2,\,b_2)$ — every weight and bias '
                            r'in one vector', va='center', fontsize=11,
             color='dimgray')
    fig.savefig(os.path.join(FIG, 'fig_nlp_statement.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote figures/fig_nlp_statement.png')


def network_diagram():
    n1, n2, n_tot = _counts()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    fig.patch.set_facecolor('white')

    x_in, x_hid, x_out = 1.5, 5.0, 8.5
    y_in = y_out = 5.3
    ys = np.linspace(8.3, 2.3, H)

    # panels behind each layer
    ax.add_patch(FancyBboxPatch((0.6, 4.5), 1.8, 1.6, boxstyle='round,pad=0.12',
                                fc='#eef7ee', ec=GREEN, lw=1.4))
    ax.add_patch(FancyBboxPatch((4.1, 1.8), 1.8, 7.0, boxstyle='round,pad=0.12',
                                fc='#f4eefb', ec=PURPLE, lw=1.4))
    ax.add_patch(FancyBboxPatch((7.6, 4.5), 1.8, 1.6, boxstyle='round,pad=0.12',
                                fc='#eef7ee', ec=GREEN, lw=1.4))

    # edges
    for y in ys:
        ax.add_patch(FancyArrowPatch((x_in + 0.42, y_in), (x_hid - 0.42, y),
                                     arrowstyle='-|>', mutation_scale=9,
                                     color='#555', lw=0.8, shrinkA=0, shrinkB=0))
        ax.add_patch(FancyArrowPatch((x_hid + 0.42, y), (x_out - 0.42, y_out),
                                     arrowstyle='-|>', mutation_scale=9,
                                     color='#555', lw=0.8, shrinkA=0, shrinkB=0))

    # nodes
    ax.add_patch(Circle((x_in, y_in), 0.42, fc='white', ec=GREEN, lw=2))
    ax.text(x_in, y_in, '$x$', ha='center', va='center', fontsize=15)
    for y in ys:
        ax.add_patch(Circle((x_hid, y), 0.36, fc='#d7c3ef', ec=PURPLE, lw=1.6))
        ax.text(x_hid, y, 'tanh', ha='center', va='center', fontsize=8)
    ax.add_patch(Circle((x_out, y_out), 0.42, fc='white', ec=GREEN, lw=2))
    ax.text(x_out, y_out, r'$\hat{y}$', ha='center', va='center', fontsize=15)

    # layer captions (all ABOVE their panels, nothing overlapping)
    ax.text(x_in, 6.75, 'Input layer', ha='center', fontsize=12, color=GREEN,
            fontweight='bold')
    ax.text(x_in, 6.32, '1 node: the input $x$', ha='center', fontsize=9,
            color='dimgray')
    ax.text(x_hid, 9.62, f'Hidden layer — {H} tanh neurons', ha='center',
            fontsize=12, color=PURPLE, fontweight='bold')
    ax.text(x_hid, 9.18, 'each neuron bends the curve in one place',
            ha='center', fontsize=8.5, color='dimgray')
    ax.text(x_out, 6.75, 'Output layer', ha='center', fontsize=12, color=GREEN,
            fontweight='bold')
    ax.text(x_out, 6.32, '1 node: the prediction', ha='center', fontsize=9,
            color='dimgray')

    # parameter blocks, in the gaps between panels (counts computed, not typed)
    ax.text(2.75, 1.30, r'$W_1,\;b_1$', ha='center', fontsize=13, color=NAVY,
            fontweight='bold')
    ax.text(2.75, 0.80, f'({H}×1) + ({H})', ha='center', fontsize=9, color='dimgray')
    ax.text(2.75, 0.40, f'= {n1} parameters', ha='center', fontsize=9, color='dimgray')
    ax.text(7.25, 1.30, r'$W_2,\;b_2$', ha='center', fontsize=13, color=NAVY,
            fontweight='bold')
    ax.text(7.25, 0.80, f'(1×{H}) + (1)', ha='center', fontsize=9, color='dimgray')
    ax.text(7.25, 0.40, f'= {n2} parameters', ha='center', fontsize=9, color='dimgray')
    ax.text(5.0, 0.75,
            f'{n1} + {n2} = {n_tot}', ha='center', fontsize=13, color=NAVY,
            fontweight='bold')
    ax.text(5.0, 0.30, r'numbers in ONE vector $w\in\mathbb{R}^{%d}$' % n_tot,
            ha='center', fontsize=9.5, color=NAVY)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_network.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote figures/fig_network.png')


if __name__ == '__main__':
    nlp_statement()
    network_diagram()
