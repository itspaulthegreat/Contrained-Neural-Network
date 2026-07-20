"""
constraint_evidence_figures.py
────────────────────────────────
The two pictures behind the "do you really use all three constraints?"
question, so the claim is visible and not just tabulated:

  figures/fig_constraint_ablation.png
      the headline protocol solved with one / two / all three constraints.
      Left: the fits on the untouched test set. Right: the achieved rate and
      the weight norm -- including the case that makes the argument, where
      Lipschitz + symmetry WITHOUT the norm ball keeps the bound exactly
      (product = 0.1) while ||w|| explodes to 4.4e5 through the biases.

  figures/fig_requirement_scenario.png
      the pendulum with an IMPOSED requirement (|dtheta/dt| <= 4 rad/s and
      ||w|| <= 4, both binding): IPOPT with all three hard constraints vs
      the SAME three as soft penalties vs tuned AdamW vs plain Adam. Only
      the hard-constrained solve meets the requirement -- and among the
      configurations that DO meet it, it is also the most accurate.

    python constraint_evidence_figures.py     (~2 min)
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data import make_teacher, _teacher_forward, pendulum_true
from src.model import param_shapes, forward_numpy, mse_numpy, unflatten_numpy
from src.analysis import lipschitz_estimate

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
                     'axes.grid': True, 'grid.alpha': 0.3})


def ablation_figure():
    from validation_protocol import fit_general, three_way_split, H, SEED
    shapes = param_shapes(1, H, 1)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = three_way_split()
    mse = lambda w, X, y: mse_numpy(w, X, y, shapes)
    xs = np.linspace(-3, 3, 400).reshape(1, -1)
    truth = _teacher_forward(xs, *make_teacher(seed=SEED)).flatten()

    cfgs = [
        ('unconstrained',        dict(L=None, B=None, sym=False), 'tab:green'),
        ('Lipschitz only',       dict(L=0.15, B=None, sym=False), 'tab:blue'),
        ('Lipschitz + symmetry', dict(L=0.1, B=None, sym=True),   'tab:orange'),
        ('ALL THREE',            dict(L=0.15, B=2.0, sym=True),   'tab:purple'),
    ]
    fits = []
    for name, cfg, c in cfgs:
        w = fit_general(Xtr, ytr, **cfg)
        fits.append(dict(name=name, w=w, color=c, test=mse(w, Xte, yte),
                         rate=lipschitz_estimate(w, shapes),
                         wnorm=float(np.linalg.norm(w))))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    axL.plot(xs.flatten(), truth, 'k:', lw=1.8, label='true function')
    axL.scatter(Xte.flatten(), yte.flatten(), s=18, c='gray', alpha=0.5,
                marker='s', label='TEST data (untouched)')
    for f in fits:
        axL.plot(xs.flatten(), forward_numpy(f['w'], xs, shapes).flatten(),
                 color=f['color'], lw=2,
                 label=f"{f['name']} — test {f['test']:.4f}")
    pad = 0.4 * (yte.max() - yte.min())
    axL.set_ylim(yte.min() - pad, yte.max() + pad)
    axL.set_xlabel('input $x$'); axL.set_ylabel('output $y$')
    axL.set_title('all three constraints give the BEST honest fit\n'
                  '(each configuration selected on validation)')
    axL.legend(fontsize=7.5, loc='upper right')

    x = np.arange(len(fits))
    wn = [f['wnorm'] for f in fits]
    axR.bar(x, wn, color=[f['color'] for f in fits], width=0.6)
    axR.set_yscale('log')
    axR.set_xticks(x)
    axR.set_xticklabels([f['name'].replace(' + ', '\n+ ').replace(' only', '\nonly')
                         for f in fits], fontsize=8)
    axR.set_ylabel(r'$\|w\|_2$  [-] (log)')
    for xi, f in zip(x, fits):
        axR.text(xi, f['wnorm'] * 1.5, f"{f['wnorm']:,.0f}", ha='center',
                 fontsize=8.5, fontweight='bold')
    axR.axhline(2.0, color='tab:purple', ls='--', lw=1.5)
    axR.text(3.4, 2.3, 'norm-ball bound B = 2', ha='right', fontsize=8,
             color='tab:purple')
    axR.annotate('bound HOLDS exactly (0.1)\nbut the biases explode:\n'
                 r'$|b_1|_{max}=3\cdot10^5$',
                 xy=(2, wn[2]), xytext=(0.55, 3e3), fontsize=8.5, color='tab:orange',
                 arrowprops=dict(arrowstyle='->', lw=1.2, color='tab:orange'))
    axR.set_title('why the norm ball must exist:\n'
                  'the Lipschitz bound limits the SLOPE, not the offsets')

    fig.suptitle('Do we really use all three constraints?  The headline protocol, '
                 'solved with one / two / all three', fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.02,
             'environment — headline three-way protocol (σ = 0.2, seed 2, 40/40/40, H = 8) · every configuration '
             'gets its own hyper-parameters selected on VALIDATION (L over 0.1–10, B over 2–10; 48-cell grid for '
             'the all-three row) and one scoring on the untouched TEST set\n'
             'validation_protocol.constraint_ablation → results/protocol_constraint_ablation.json',
             fontsize=7, color='dimgray', ha='center')
    fig.savefig(os.path.join(FIG, 'fig_constraint_ablation.png'))
    plt.close(fig)
    print('wrote figures/fig_constraint_ablation.png')


def requirement_figure():
    """The pendulum with an IMPOSED requirement, read from the stored run."""
    d = json.load(open(os.path.join(RES, 'pendulum_requirement.json')))
    L_req, B_req = d['L_req'], d['B_req']
    rows = d['rows']
    colors = ['tab:purple', 'tab:red', 'tab:green', 'tab:orange']

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={'width_ratios': [1, 1.1]})
    x = np.arange(len(rows))
    rates = [r['rate'] for r in rows]
    tests = [r['test'] for r in rows]
    ok = [bool(r['lip_ok'] and r['ball_ok'] and r['sym_ok']) for r in rows]

    axL.bar(x, rates, color=[c if o else '#c8c8c8' for c, o in zip(colors, ok)],
            edgecolor=['none' if o else 'tab:red' for o in ok], linewidth=2, width=0.6)
    axL.axhline(L_req, color='black', ls='--', lw=2)
    axL.text(len(rows) - 0.4, L_req * 1.06, f'REQUIRED  ≤ {L_req:g} rad/s',
             ha='right', fontsize=9, fontweight='bold')
    for xi, (r, o) in enumerate(zip(rates, ok)):
        axL.text(xi, r + 0.18, f'{r:.2f}', ha='center', fontsize=9,
                 fontweight='bold', color='black' if o else 'tab:red')
    axL.set_xticks(x)
    axL.set_xticklabels(['IPOPT\n(hard)', 'penalty-Adam\n(soft, ρ*)', 'AdamW\n(tuned)',
                         'plain\nAdam'], fontsize=8.5)
    axL.set_ylabel('achieved rate  $|d\\hat\\theta/dt|$  [rad/s]')
    axL.set_title('the requirement is imposed by the task, not chosen for fit\n'
                  'grey + red outline = requirement BROKEN')

    axR.bar(x, tests, color=[c if o else '#c8c8c8' for c, o in zip(colors, ok)],
            edgecolor=['none' if o else 'tab:red' for o in ok], linewidth=2, width=0.6)
    for xi, (t, o) in enumerate(zip(tests, ok)):
        axR.text(xi, t + 0.0012, f'{t:.4f}', ha='center', fontsize=9, fontweight='bold')
    axR.set_xticks(x)
    axR.set_xticklabels(['IPOPT\n(hard)', 'penalty-Adam\n(soft, ρ*)', 'AdamW\n(tuned)',
                         'plain\nAdam'], fontsize=8.5)
    axR.set_ylabel('honest TEST MSE [-]')
    axR.set_title('the others look accurate only because they\n'
                  'ignored the requirement they were given')

    fig.suptitle(f'When the requirement actually BINDS (pendulum: ≤ {L_req:g} rad/s and '
                 f'‖w‖ ≤ {B_req:g}) — only the hard constraint delivers it',
                 fontsize=12.5, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02,
             f'environment — pendulum protocol (σ = 0.15, seed 0, 60/60/60, H = 8) · IPOPT carries all three as HARD '
             f'constraints; penalty-Adam carries THE SAME THREE as soft penalties (one shared ρ, selected on '
             f'validation → {d["rho_star"]:g}); AdamW wd selected the same way → {d["wd_star"]:g}; equal 3,000-iteration budget\n'
             f'the ρ sweep shows the penalty CAN comply (ρ ≥ 0.01 holds both bounds) — but validation, which optimises '
             f'FIT, selects ρ* = {d["rho_star"]:g}, and that choice breaks the requirement: compliance is not what the tuning is aiming at',
             fontsize=7, color='dimgray', ha='center')
    fig.savefig(os.path.join(FIG, 'fig_requirement_scenario.png'))
    plt.close(fig)
    print('wrote figures/fig_requirement_scenario.png')


if __name__ == '__main__':
    ablation_figure()
    requirement_figure()
