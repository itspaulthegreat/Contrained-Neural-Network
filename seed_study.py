"""
seed_study.py
───────────────
Statistical honesty check: is IPOPT's edge (or Adam's apparent edge at
sigma=0.1) real, or within seed noise? And is the Adam baseline FAIR --
i.e. Adam WITH regularization (weight decay), not just plain Adam?

For each noise level sigma in {0.1, 0.3}, across N random seeds (each seed
draws a fresh noisy train/test split AND a fresh initial guess), we run
three methods on the identical data:

  - IPOPT   : hard Lipschitz (L_max=4) + norm-ball + symmetry  (the project)
  - Adam    : plain, unregularized                              (weak baseline)
  - AdamW   : Adam + decoupled weight decay, strength tuned on seed 0
              across a small grid so the baseline gets its BEST shot
              (steelmanned "regularized Adam")

Reports per (sigma, method): mean test MSE, std, and the head-to-head win
rate of IPOPT vs the BEST Adam variant. Also records the achieved
sensitivity ||W1||_F||W2||_F of each method -- to show that even regularized
Adam does not let you CHOOSE the sensitivity, it only nudges it.

    python seed_study.py            # ~1-2 min, writes figures/fig_seed_study.png

No conclusions are hard-coded: the verdict text is computed from the runs.
"""

import copy
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from configs.experiments import _make
from src.data import generate_dataset
from src.model import param_shapes, random_init, mse_numpy
from src.solver import solve
from src.baseline_adam import adam_optimize
from src.analysis import lipschitz_estimate

FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
N_SEEDS = 15
SIGMAS = [0.1, 0.3]
WD_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]        # weight-decay candidates

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'axes.grid': True, 'grid.alpha': 0.3})


def data_for(sigma, seed):
    return generate_dataset(n_train=60, n_test=40, noise_std=sigma, seed=seed,
                            H_teacher=6, x_range=(-3.0, 3.0))


def run_ipopt(sigma, seed):
    exp = _make(f'seedstudy_ipopt_s{seed}', 'seed study', 'seedstudy',
                method='ipopt', H=8, use_lipschitz=True, use_norm_ball=True,
                use_symmetry_break=True, L_max=4.0, B_max=6.0,
                noise_std=sigma, seed=seed)
    Xtr, ytr, Xte, yte = data_for(sigma, seed)
    r = solve(exp, Xtr, ytr, Xte, yte)
    return r['test_mse'], r['lipschitz_estimate']


def run_adam(sigma, seed, weight_decay=0.0):
    shapes = param_shapes(1, 8, 1)
    Xtr, ytr, Xte, yte = data_for(sigma, seed)
    w0 = random_init(shapes, scale=0.5, seed=seed)
    out = adam_optimize(w0, shapes, Xtr, ytr, lr=0.02, n_iter=3000,
                        weight_decay=weight_decay)
    return mse_numpy(out['w'], Xte, yte, shapes), lipschitz_estimate(out['w'], shapes)


def tune_wd(sigma):
    """Pick the weight decay minimizing test MSE on seed 0 -- Adam's best shot."""
    best_wd, best_mse = 0.0, run_adam(sigma, 0, 0.0)[0]
    for wd in WD_GRID:
        mse = run_adam(sigma, 0, wd)[0]
        if mse < best_mse:
            best_wd, best_mse = wd, mse
    return best_wd


def main():
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(11, 4.8), sharey=False)
    verdicts = []

    for ax, sigma in zip(axes, SIGMAS):
        wd = tune_wd(sigma)
        rows = {'IPOPT (hard constraint)': [], 'Adam (plain)': [],
                f'AdamW (wd={wd:g})': []}
        lips = {k: [] for k in rows}
        for seed in range(N_SEEDS):
            mi, li = run_ipopt(sigma, seed)
            ma, la = run_adam(sigma, seed, 0.0)
            mw, lw = run_adam(sigma, seed, wd)
            rows['IPOPT (hard constraint)'].append(mi); lips['IPOPT (hard constraint)'].append(li)
            rows['Adam (plain)'].append(ma);            lips['Adam (plain)'].append(la)
            rows[f'AdamW (wd={wd:g})'].append(mw);       lips[f'AdamW (wd={wd:g})'].append(lw)

        ipopt = np.array(rows['IPOPT (hard constraint)'])
        best_adam_name = min(['Adam (plain)', f'AdamW (wd={wd:g})'],
                             key=lambda k: np.mean(rows[k]))
        best_adam = np.array(rows[best_adam_name])
        ipopt_wins = int(np.sum(ipopt < best_adam))
        # paired difference + simple significance proxy (std of the mean)
        diff = best_adam - ipopt                       # >0 means IPOPT better
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        tstat = diff.mean() / se if se > 0 else float('inf')

        # honest verdict text, computed
        who = 'IPOPT' if diff.mean() > 0 else best_adam_name
        margin = abs(diff.mean())
        sig = 'clearly (|t|>2)' if abs(tstat) > 2 else 'but within seed noise (|t|<2)'
        verdicts.append(
            f"sigma={sigma}: best Adam = {best_adam_name}; "
            f"IPOPT mean {ipopt.mean():.4f}+/-{ipopt.std(ddof=1):.4f}, "
            f"{best_adam_name} {best_adam.mean():.4f}+/-{best_adam.std(ddof=1):.4f}; "
            f"IPOPT wins {ipopt_wins}/{N_SEEDS}; better = {who} {sig} "
            f"(mean gap {margin:.4f}, t={tstat:.2f}). "
            f"sensitivity: IPOPT {np.mean(lips['IPOPT (hard constraint)']):.2f} (capped at 4), "
            f"Adam {np.mean(lips['Adam (plain)']):.1f}, "
            f"AdamW {np.mean(lips[f'AdamW (wd={wd:g})']):.1f} (uncontrolled)")

        # strip + box plot
        names = list(rows.keys())
        colors = ['tab:purple', 'tab:green', 'tab:olive']
        for i, (name, c) in enumerate(zip(names, colors)):
            y = rows[name]
            x = np.random.default_rng(i).normal(i, 0.06, size=len(y))
            ax.scatter(x, y, color=c, alpha=0.7, s=28, zorder=3)
            ax.hlines(np.mean(y), i - 0.25, i + 0.25, color=c, lw=2.5, zorder=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=8)
        ax.set_ylabel('test MSE [-]')
        ax.set_title(f'$\\sigma$ = {sigma}   (IPOPT wins {ipopt_wins}/{N_SEEDS} seeds)')

    fig.suptitle(f'Seed study — {N_SEEDS} random noise draws per method '
                 f'(bar = mean; Adam given its best tuned weight decay)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURES, 'fig_seed_study.png')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)
    print('=' * 70)
    for v in verdicts:
        print(v)
        print('-' * 70)


if __name__ == '__main__':
    main()
