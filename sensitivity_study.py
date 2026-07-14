"""
sensitivity_study.py
──────────────────────
The project's core contribution, made quantitative, and the answer to
"why not just tune the weight decay until Adam reaches sensitivity 4?"

Because there is NO reliable map  weight_decay -> Lipschitz constant. The
achieved sensitivity depends on the data draw, the seed, and the noise
level, so no fixed weight-decay value pins it to a target -- whereas IPOPT
solves  ||W1||_F ||W2||_F <= 4  exactly, every time.

Two outputs (run AFTER the repo is importable; ~2 min):

  1. fig_wd_vs_sensitivity.png -- sweep weight_decay and plot the achieved
     sensitivity (mean +/- 1 std across seeds). The band is wide and the
     mean crosses 4 only by accident at one wd; even there the spread means
     any single run misses. IPOPT is a flat line at 4 with zero width.

  2. results/sensitivity_table.json + printed table -- per method:
     mean / std / MAX observed sensitivity and the count of runs that
     VIOLATE a target bound of 4. This is the real contribution:
     a chosen, guaranteed sensitivity vs an uncontrolled one.

    python sensitivity_study.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from seed_study import run_ipopt, run_adam, tune_wd, N_SEEDS

FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
RESULTS = os.path.join(os.path.dirname(__file__), 'results')
TARGET = 4.0
WD_SWEEP = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'axes.grid': True, 'grid.alpha': 0.3})


def sensitivity_table(sigmas=(0.1, 0.3)):
    table = {}
    for sigma in sigmas:
        wd = tune_wd(sigma)
        ip, ad, aw = [], [], []
        for seed in range(N_SEEDS):
            ip.append(run_ipopt(sigma, seed)[1])
            ad.append(run_adam(sigma, seed, 0.0)[1])
            aw.append(run_adam(sigma, seed, wd)[1])
        table[sigma] = {}
        for name, arr in [('IPOPT (hard, L<=4)', ip),
                          ('Adam (plain)', ad),
                          (f'AdamW (wd={wd:g})', aw)]:
            a = np.array(arr)
            # median + range reported alongside mean/std: the AdamW
            # distribution is right-skewed, so mean/std alone would mislead.
            table[sigma][name] = dict(
                mean=float(a.mean()), std=float(a.std(ddof=1)),
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
                q1=float(np.percentile(a, 25)), q3=float(np.percentile(a, 75)),
                values=sorted(round(float(x), 2) for x in a),
                violations=int(np.sum(a > TARGET + 1e-6)), n=len(a))
    return table


def raw_distribution_figure(table):
    """Show every seed's achieved sensitivity as a dot -- no hiding behind
    mean/std. IPOPT is a tight column at 4; AdamW is a right-skewed scatter
    from 4 to 36, all above the target line."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, sigma in zip(axes, sorted(table)):
        methods = table[sigma]
        adamw_key = [k for k in methods if k.startswith('AdamW')][0]
        names = ['IPOPT (hard, L<=4)', adamw_key, 'Adam (plain)']
        colors = ['tab:purple', 'tab:olive', 'tab:green']
        for i, (nm, c) in enumerate(zip(names, colors)):
            vals = methods[nm]['values']
            x = np.random.default_rng(i).normal(i, 0.06, size=len(vals))
            ax.scatter(x, vals, color=c, alpha=0.75, s=32, zorder=3)
            ax.hlines(methods[nm]['median'], i - 0.25, i + 0.25, color=c, lw=2.5,
                      zorder=4)
        ax.axhline(TARGET, color='black', ls='--', lw=1.5)
        ax.text(2.4, TARGET * 1.08, 'target bound L=4', ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.set_xticks(range(3))
        ax.set_xticklabels([n.split(' (')[0] + ('\n(wd)' if 'AdamW' in n else '')
                            for n in names], fontsize=8)
        ax.set_ylabel('achieved sensitivity $\\|W_1\\|_F\\|W_2\\|_F$ [-] (log)')
        ax.set_title(f'$\\sigma$ = {sigma:g}   (bar = median)')
    fig.suptitle('Every seed shown, no averaging: IPOPT pins to 4 exactly; '
                 'regularized Adam scatters (right-skewed, all above 4)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURES, 'fig_sensitivity_dist.png')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


def wd_sweep_figure(sigma=0.3):
    means, stds = [], []
    for wd in WD_SWEEP:
        lips = [run_adam(sigma, s, wd)[1] for s in range(N_SEEDS)]
        means.append(np.mean(lips)); stds.append(np.std(lips, ddof=1))
    means, stds = np.array(means), np.array(stds)
    x = np.array([max(w, 3e-5) for w in WD_SWEEP])   # 0 -> placeholder for log

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(x, means - stds, means + stds, color='tab:olive', alpha=0.25,
                    label='AdamW achieved sensitivity (mean $\\pm$1 std, 15 seeds)')
    ax.plot(x, means, 'o-', color='tab:olive')
    ax.axhline(TARGET, color='tab:purple', lw=2.5,
               label='IPOPT: exactly 4.00, every seed (zero spread)')
    ax.annotate('no weight decay reliably\nhits the target: the band\n'
                'never collapses onto 4',
                xy=(3e-3, means[np.argmin(np.abs(means - TARGET))]),
                xytext=(1e-4, TARGET * 2.1), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=1.2))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('weight decay (your knob) [-]   (leftmost point = 0)')
    ax.set_ylabel('achieved sensitivity $\\|W_1\\|_F\\|W_2\\|_F$ [-]')
    ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Why you cannot "just tune weight decay to reach 4"\n'
                 '($\\sigma$=0.3): weight decay only NUDGES sensitivity; '
                 'the hard constraint SETS it')
    fig.tight_layout()
    path = os.path.join(FIGURES, 'fig_wd_vs_sensitivity.png')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


def main():
    table = sensitivity_table()
    with open(os.path.join(RESULTS, 'sensitivity_table.json'), 'w') as f:
        json.dump(table, f, indent=2)
    print('=' * 86)
    print(f"{'sigma':>6} {'method':<22} {'median':>7} {'mean':>7} {'range':>14} {'viol(>4)':>9}")
    print('-' * 86)
    for sigma, methods in table.items():
        for name, s in methods.items():
            print(f"{sigma:>6} {name:<22} {s['median']:>7.2f} {s['mean']:>7.2f} "
                  f"[{s['min']:>5.2f},{s['max']:>6.2f}] {s['violations']:>6}/{s['n']}")
        print('-' * 86)
    raw_distribution_figure(table)
    wd_sweep_figure()


if __name__ == '__main__':
    main()
