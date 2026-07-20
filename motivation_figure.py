"""
motivation_figure.py
──────────────────────
THE PROBLEM, shown before the solution. Two panels under ONE consistent
pair of reference values (the bound L = 4 and the noise levels shown),
built entirely from stored results (run AFTER `python main.py` and
`python sensitivity_study.py`):

Left  — in standard (unconstrained) training, the sensitivity
        ||W1||_F ||W2||_F is whatever the data and the optimizer make it:
        12x growth with label noise (Adam, sigma 0 -> 0.3), 93x with model
        size (GN, H 4 -> 64), drifting straight through the default bound
        L = 4 that the seed study certifies.

Right — the STANDARD workaround, regularization you tune (AdamW weight
        decay), swept at sigma = 0.3 (the left panel's own stress bar)
        against the SAME bound 4: the achieved-sensitivity band never
        collapses onto the target. Tuning nudges the property; it cannot
        set it. (The penalty-in-the-loss variant is a different study at
        its own conditions and is dissected on the exact-vs-penalty slide.)

Redesigned 2026-07-18: the earlier right panel showed the hinge-penalty
sweep at sigma = 0.05 / L_max = 1 — different noise, different bound on
the opening slide — which was confusing; this version keeps every value
on the slide consistent.

    python motivation_figure.py    ->  figures/fig_motivation.png
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, 'results')
FIGURES = os.path.join(HERE, 'figures')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 11.5, 'figure.dpi': 150,
    'lines.linewidth': 2, 'savefig.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3,
})


def load(name):
    return json.load(open(os.path.join(RESULTS, name + '.json')))


def main():
    REQUIRED = 4.0   # the sensitivity level a user might need to promise

    # left panel data: unconstrained sensitivity across situations
    adam = [(f'Adam\n$\\sigma$={s}', load(f'exp_noise_{t}_adam')['lipschitz_estimate'])
            for s, t in [(0.0, '0p0'), (0.1, '0p1'), (0.2, '0p2'), (0.3, '0p3')]]
    gn = [(f'GN\nH={H}', load(f'exp_complexity_gn_H{H}')['lipschitz_estimate'])
          for H in [4, 16, 64]]


    # right panel: the weight-decay sweep at sigma = 0.3 -- the SAME noise
    # level as the left panel's stress bar and the SAME target bound L = 4 as
    # the left panel's dashed line (stored by sensitivity_study.py)
    wd = json.load(open(os.path.join(RESULTS, 'wd_sweep_sigma0p3.json')))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.0))

    labels = [a[0] for a in adam + gn]
    vals = [a[1] for a in adam + gn]
    colors = ['tab:green'] * len(adam) + ['tab:red'] * len(gn)
    x = np.arange(len(labels))
    axL.bar(x, vals, color=colors, width=0.62)
    axL.axhline(REQUIRED, color='black', ls='--', lw=2)
    axL.text(len(labels) - 0.4, REQUIRED * 1.25, 'the bound that had to be\npromised (default L = 4)',
             ha='right', fontsize=9, fontweight='bold')
    for xi, v in zip(x, vals):
        axL.text(xi, v * 1.12, f'{v:.3g}', ha='center', fontsize=9)
    axL.set_yscale('log')
    axL.set_ylim(0.5, 400)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel('sensitivity  $\\|W_1\\|_F\\|W_2\\|_F$  [-] (log)')
    axL.set_title('unconstrained training: sensitivity is\nwhatever data and size make it')

    x = np.array([max(w, 3e-5) for w in wd['wd']])   # wd = 0 -> log placeholder
    med = np.array(wd['median'])
    vmin, vmax = np.array(wd['min']), np.array(wd['max'])
    # band = OBSERVED min-max across seeds: every edge is a real run
    axR.fill_between(x, vmin, vmax, color='tab:olive', alpha=0.25,
                     label=f"AdamW: observed range over {wd['n_seeds']} seeds (min–max)")
    axR.plot(x, med, 'o-', color='tab:olive', label='median seed')
    axR.axhline(wd['target'], color='black', ls='--', lw=2)
    axR.text(x[-1], wd['target'] * 1.18, 'the same bound L = 4', ha='right',
             fontsize=9, fontweight='bold')
    axR.annotate('no weight decay PINS the value:\nthe observed range never\ncollapses onto 4',
                 xy=(3e-3, med[np.argmin(np.abs(med - wd['target']))]),
                 xytext=(1.2e-4, wd['target'] * 3.6), fontsize=9,
                 arrowprops=dict(arrowstyle='->', lw=1.2))
    axR.set_xscale('log'); axR.set_yscale('log')
    axR.set_xlabel('weight decay [-]  (leftmost point = 0; a guess)')
    axR.set_ylabel('sensitivity  $\\|W_1\\|_F\\|W_2\\|_F$  [-] (log)')
    axR.set_title('the standard fix (tuned regularization):\n'
                  'weight decay NUDGES sensitivity, never PINS it')
    axR.legend(fontsize=8, loc='upper right')

    fig.suptitle('THE PROBLEM — standard NN training cannot promise the properties a task needs',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    # (the environment note lives on the SLIDE, not inside the figure)
    path = os.path.join(FIGURES, 'fig_motivation.png')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    main()
