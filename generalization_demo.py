"""
generalization_demo.py
────────────────────────
Leo's central request: "a visualization of the function you learn, and the
difference between learning with or without the constraint. Also, check the
fit on a validation dataset."

Same teacher-student data, same student network, solved two ways:
  - WITHOUT the Lipschitz constraint: the over-parameterized student is free
    to chase the noise -> wiggly fit, good on train, worse on validation.
  - WITH the Lipschitz constraint (||W1||_F ||W2||_F <= L): the fit is forced
    smooth -> slightly worse on train, better on validation.

Produces:
  figures/fig_generalization.png   -- static side-by-side (for the report)
  figures/anim_with_without.gif    -- morph unconstrained -> constrained, live
                                       train/validation MSE (for the talk)

    python generalization_demo.py
"""

import os
import numpy as np
import casadi as ca

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim

from src.data import generate_dataset, make_teacher, _teacher_forward
from src.model import (param_shapes, n_params, forward_symbolic, forward_numpy,
                       mse_numpy, random_init, unflatten_symbolic)
from src.constraints import lipschitz_constraint

FIG = os.path.join(os.path.dirname(__file__), 'figures')
X_RANGE = (-3.0, 3.0)
SIGMA = 0.2           # enough noise that an unconstrained over-fit is visible
H = 8
L_MAX = 1.5           # tight enough to smooth the fit, loose enough to fit
SEED = 2

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'lines.linewidth': 2, 'axes.grid': True, 'grid.alpha': 0.3})


def _solve(use_lip):
    shapes = param_shapes(1, H, 1)
    Xtr, ytr, Xva, yva = generate_dataset(n_train=40, n_test=40, noise_std=SIGMA,
                                          seed=SEED, x_range=X_RANGE)
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    if use_lip:
        W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
        g, lb, ub = lipschitz_constraint(W1, W2, L_MAX)
        nlp = {'x': w, 'f': f, 'g': g}
        args = dict(lbg=lb, ubg=ub)
    else:
        nlp = {'x': w, 'f': f}
        args = {}
    solver = ca.nlpsol('s', 'ipopt', nlp,
                       dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                            print_time=False))
    sol = solver(x0=random_init(shapes, scale=0.5, seed=SEED), **args)
    w_opt = np.asarray(sol['x']).flatten()
    return dict(w=w_opt,
                train=mse_numpy(w_opt, Xtr, ytr, shapes),
                val=mse_numpy(w_opt, Xva, yva, shapes),
                Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)


def main():
    unc = _solve(use_lip=False)
    con = _solve(use_lip=True)
    shapes = param_shapes(1, H, 1)
    xs = np.linspace(*X_RANGE, 400).reshape(1, -1)
    teacher = make_teacher(seed=SEED)
    y_true = _teacher_forward(xs, *teacher).flatten()
    f_unc = forward_numpy(unc['w'], xs, shapes).flatten()
    f_con = forward_numpy(con['w'], xs, shapes).flatten()

    print(f'WITHOUT constraint: train {unc["train"]:.4f}  validation {unc["val"]:.4f}')
    print(f'WITH    constraint: train {con["train"]:.4f}  validation {con["val"]:.4f}')
    print(f'validation improvement: {(unc["val"]-con["val"])/unc["val"]*100:.0f}% lower with the constraint')

    # ---- static side-by-side (report figure) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    pad = 0.3 * (unc['ytr'].max() - unc['ytr'].min())
    for ax, res, curve, ttl, c in [
            (axes[0], unc, f_unc, 'WITHOUT constraint', 'tab:green'),
            (axes[1], con, f_con, f'WITH constraint  ($\\|W_1\\|_F\\|W_2\\|_F \\leq {L_MAX}$)', 'tab:purple')]:
        ax.plot(xs.flatten(), y_true, color='black', ls=':', lw=1.8, label='true function')
        ax.scatter(res['Xtr'].flatten(), res['ytr'].flatten(), s=20, c='tab:blue',
                   alpha=0.6, label='training data')
        ax.scatter(res['Xva'].flatten(), res['yva'].flatten(), s=22, c='tab:orange',
                   alpha=0.7, marker='^', label='validation data')
        ax.plot(xs.flatten(), curve, color=c, lw=2.5, label='learned fit')
        ax.set_ylim(unc['ytr'].min() - pad, unc['ytr'].max() + pad)
        ax.set_xlabel('input $x$')
        ax.set_title(f'{ttl}\ntrain MSE {res["train"]:.4f}   validation MSE {res["val"]:.4f}',
                     fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('output $y$')
    fig.suptitle('Why the Lipschitz constraint helps: the same over-parameterized network,\n'
                 'learned without vs with the constraint (validation improves)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02,
             f'environment — σ = {SIGMA}, seed {SEED}, 40 train / 40 held-out, H = 8 · constrained run: '
             f'Lipschitz ONLY, L = {L_MAX}\n'
             'illustration twin of the headline protocol (which additionally holds out an untouched test set)',
             fontsize=7, color='dimgray', ha='center')
    fig.savefig(os.path.join(FIG, 'fig_generalization.png'))
    plt.close(fig)
    print('wrote figures/fig_generalization.png')

    # ---- morph animation (talk) ----
    n_hold, n_morph = 12, 30
    frames = n_hold + n_morph + n_hold
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_xlim(*X_RANGE); ax.set_ylim(unc['ytr'].min() - pad, unc['ytr'].max() + pad)
    ax.plot(xs.flatten(), y_true, color='black', ls=':', lw=1.8, label='true function')
    ax.scatter(unc['Xtr'].flatten(), unc['ytr'].flatten(), s=20, c='tab:blue', alpha=0.6, label='training data')
    ax.scatter(unc['Xva'].flatten(), unc['yva'].flatten(), s=22, c='tab:orange', alpha=0.7, marker='^', label='validation data')
    line, = ax.plot([], [], lw=2.8)
    ax.set_xlabel('input $x$'); ax.set_ylabel('output $y$')
    ax.legend(fontsize=8, loc='upper right')
    ttl = ax.set_title('')

    def update(k):
        if k < n_hold:
            a = 0.0
        elif k < n_hold + n_morph:
            a = (k - n_hold + 1) / n_morph
        else:
            a = 1.0
        y = (1 - a) * f_unc + a * f_con
        mse_v = (1 - a) * unc['val'] + a * con['val']
        line.set_data(xs.flatten(), y)
        line.set_color((0.17 + 0.24 * a, 0.5 - 0.3 * a, 0.15 + 0.45 * a))
        state = 'WITHOUT constraint' if a < 0.5 else 'WITH constraint'
        line.set_label(state)
        ttl.set_text(f'{state}   ·   validation MSE {mse_v:.4f}'
                     + ('   (chasing noise)' if a < 0.3 else
                        '   (smooth, generalizes)' if a > 0.7 else '   (turning on the constraint...)'))
        return line, ttl

    anim = manim.FuncAnimation(fig, update, frames=frames, blit=False)
    fig.tight_layout()
    anim.save(os.path.join(FIG, 'anim_with_without.gif'),
              writer=manim.PillowWriter(fps=12))
    plt.close(fig)
    print('wrote figures/anim_with_without.gif')


if __name__ == '__main__':
    main()
