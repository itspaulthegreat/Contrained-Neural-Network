"""
motivation_figure.py
──────────────────────
THE PROBLEM, shown before the solution. Two panels, built entirely from
stored results (run AFTER `python main.py`):

Left  — in standard (unconstrained) training, the model's sensitivity
        ||W1||_F ||W2||_F is whatever the data and the optimizer make it:
        it grows 12x with label noise (Adam) and 93x with model size (GN),
        and nothing the user chose controls it. If you needed to PROMISE a
        sensitivity bound (robustness, safety, physical plausibility),
        standard training cannot make that promise.

Right — the standard workaround, a penalty term rho * violation in the
        loss, is a guessing game: too small and the requirement is simply
        ignored; the entire ignored->enforced transition hides inside a
        hairline window of the hyperparameter rho, found only by search,
        and even a zero measured violation is a post-hoc observation, not
        a guarantee.

This is the pain the project addresses: hard constraints make the promise
part of the problem statement, and the solver keeps it by construction.

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
            for s, t in [(0.0, '0p0'), (0.1, '0p1'), (0.3, '0p3')]]
    gn = [(f'GN\nH={H}', load(f'exp_complexity_gn_H{H}')['lipschitz_estimate'])
          for H in [4, 16, 64]]

    # right panel data: penalty violation vs rho
    pen = []
    for f in os.listdir(RESULTS):
        if f.startswith('exp_penalty_rho'):
            r = json.load(open(os.path.join(RESULTS, f)))
            pen.append((r['rho'], r['max_constraint_violation']))
    pen.sort()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axL, axR = axes

    labels = [a[0] for a in adam + gn]
    vals = [a[1] for a in adam + gn]
    colors = ['tab:green'] * len(adam) + ['tab:red'] * len(gn)
    x = np.arange(len(labels))
    axL.bar(x, vals, color=colors, width=0.62)
    axL.axhline(REQUIRED, color='black', ls='--', lw=2)
    axL.text(len(labels) - 0.4, REQUIRED * 1.25, 'the bound you\nneeded to promise',
             ha='right', fontsize=9, fontweight='bold')
    for xi, v in zip(x, vals):
        axL.text(xi, v * 1.12, f'{v:.3g}', ha='center', fontsize=9)
    axL.set_yscale('log')
    axL.set_ylim(0.5, 400)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel('sensitivity  $\\|W_1\\|_F\\|W_2\\|_F$  [-] (log)')
    axL.set_title('unconstrained training: sensitivity is\nwhatever data and size make it')

    rho_eps = 3e-6
    rho = [max(r, rho_eps) for r, _ in pen]
    vio = [v for _, v in pen]
    axR.plot(rho, vio, 'o-', color='tab:red')
    axR.axhline(0.0, color='tab:purple', ls='--', lw=2)
    axR.text(2e-3, 0.06, 'requirement: violation = 0', color='tab:purple',
             fontsize=9, fontweight='bold')
    axR.axvspan(1e-5, 1e-4, color='tab:orange', alpha=0.18)
    axR.annotate('requirement\nsimply ignored', xy=(rho_eps * 1.6, 1.42), fontsize=9, ha='left')
    axR.annotate('the ENTIRE usable range\nof the guess $\\rho$',
                 xy=(3e-5, 0.75), xytext=(3e-3, 1.1), fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', lw=1.2))
    axR.set_xscale('log')
    axR.set_xlabel('penalty weight $\\rho$ [-] (your guess)')
    axR.set_ylabel('constraint violation [-]')
    axR.set_title('the standard fix (penalty in the loss):\nenforcement is a hyperparameter guess')

    fig.suptitle('THE PROBLEM — standard NN training cannot promise the properties you need',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURES, 'fig_motivation.png')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    main()
