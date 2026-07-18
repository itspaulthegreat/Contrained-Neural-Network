"""
robustness_demo.py
────────────────────
The real-world payoff of the Lipschitz constraint: a CERTIFIED robustness
bound. For the constrained network the training guarantee

    |f(x + d) - f(x)|  <=  L_max * |d|      for ALL x and ALL d

holds by construction (tanh is 1-Lipschitz, so Lip(f) <= ||W1||_F ||W2||_F
<= L_max). This is exactly the property the certified-robustness literature
wants from a network facing sensor errors or adversarial input perturbations
(cf. spectral normalization / Lipschitz-constrained training).

This script takes the ALREADY-TRAINED weights from the noise study
(sigma = 0.3: exp_noise_0p3_ipopt, L_max = 4, vs exp_noise_0p3_adam) and
measures the realized worst-case output change

    W(eps) = max over x, x' in [-3, 3] with |x - x'| <= eps of |f(x) - f(x')|

on a fine grid, then plots it against the certificate line L_max * eps.
No re-training, no new data -- pure analysis of existing results.

Run AFTER `python main.py` (needs results/exp_noise_0p3_*.json):

    python robustness_demo.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.model import param_shapes, forward_numpy

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
FIGURES = os.path.join(os.path.dirname(__file__), 'figures')

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 150,
    'lines.linewidth': 2, 'savefig.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3,
})


def worst_case_change(w, H, eps_list, x_range=(-3.0, 3.0), n_grid=3001):
    """Exact-on-grid modulus of continuity: for each eps, the largest output
    change any input perturbation up to eps can cause, anywhere in x_range."""
    shapes = param_shapes(1, H, 1)
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    f = forward_numpy(np.asarray(w), xs.reshape(1, -1), shapes).flatten()
    dx = xs[1] - xs[0]
    out = []
    for eps in eps_list:
        k = max(1, int(round(eps / dx)))          # window radius in grid steps
        # max over all windows of (max f - min f) within distance eps
        w_max = np.array([f[i:i + k + 1].max() for i in range(len(f) - k)])
        w_min = np.array([f[i:i + k + 1].min() for i in range(len(f) - k)])
        out.append(float((w_max - w_min).max()))
    return np.array(out)


def main():
    con = json.load(open(os.path.join(RESULTS, 'exp_noise_0p3_ipopt.json')))
    ada = json.load(open(os.path.join(RESULTS, 'exp_noise_0p3_adam.json')))
    L = con['L_max']

    L_adam = ada['lipschitz_estimate']            # a-posteriori, uncontrolled
    eps = np.linspace(0.0, 0.3, 31)[1:]
    wc_con = worst_case_change(con['w'], con['H'], eps)
    wc_ada = worst_case_change(ada['w'], ada['H'], eps)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    # guaranteed ceilings -- what a safety engineer must budget against
    ax.plot(eps, L * eps, ls='--', color='tab:purple', lw=2.2,
            label=f'constrained ceiling: $L_{{max}}\\varepsilon = {L:g}\\varepsilon$ — CHOSEN before training')
    ax.plot(eps, L_adam * eps, ls='--', color='tab:green', lw=2.2,
            label=f'Adam ceiling: ${L_adam:.1f}\\varepsilon$ — discovered AFTER training, uncontrolled')
    ax.fill_between(eps, L * eps, L_adam * eps, color='tab:green', alpha=0.10)
    # realized worst cases (grid lower bounds -- both respect their ceilings)
    ax.plot(eps, wc_con, color='tab:purple', lw=2,
            label='constrained — realized worst case')
    ax.plot(eps, wc_ada, color='tab:green', lw=2,
            label='Adam — realized worst case')
    ax.annotate('the safety budget gap:\nAdam forces you to design\n'
                f'against a {L_adam / L:.1f}x higher ceiling',
                xy=(0.22, (L + L_adam) / 2 * 0.22), fontsize=9,
                ha='center', color='#333333')

    ax.set_xlabel('input perturbation size $\\varepsilon$ [-]')
    ax.set_ylabel('worst-case output change [-]')
    ax.set_xlim(0, 0.3)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Certified robustness — the guaranteed ceiling is chosen, '
                 'not discovered\n(trained nets from the noise study, '
                 '$\\sigma$ = 0.3; bound holds for every $x$, every perturbation)',
                 fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02,
             'environment — pure analysis of STORED weights from Group 4 (σ = 0.3, 60 train / 40 test, '
             'H = 8): IPOPT (L = 4 + ball 6 + symmetry) vs plain Adam (unconstrained), no retraining\n'
             'certificate = the a-priori bound; realized = worst case actually measured over x ∈ [−3, 3]',
             fontsize=7, color='dimgray', ha='center')
    path = os.path.join(FIGURES, 'fig_robustness.png')
    fig.savefig(path)
    plt.close(fig)

    print(f'wrote {path}')
    print(f'constrained: certificate {L:g}*eps, realized slope ~ '
          f'{wc_con[-1] / eps[-1]:.2f} (always <= {L:g}, guaranteed)')
    print(f'Adam:        no certificate, realized slope ~ '
          f'{wc_ada[-1] / eps[-1]:.2f}, weight-norm bound '
          f'{ada["lipschitz_estimate"]:.1f}')

    # the true Lipschitz constant sup|f'(x)|, analytically:
    # f'(x) = sum_j W2_j * tanh'(z_j) * W1_j. The gap between this and the
    # norm-product certificate has two sources: neurons peak (tanh' = 1) at
    # DIFFERENT x, and mixed weight signs partially cancel (Cauchy-Schwarz
    # equality needs perfect alignment). The certificate must dominate the
    # worst case over all inputs and alignments -- slack is the price of an
    # unconditional promise. (Exact Lipschitz constants of general networks
    # are NP-hard, Virmaux & Scaman 2018; norm products are the tractable
    # certificate.)
    from src.model import unflatten_numpy
    for tag, r in (('constrained', con), ('Adam', ada)):
        W1, b1, W2, b2 = unflatten_numpy(np.asarray(r['w']), param_shapes(1, r['H'], 1))
        xs = np.linspace(-6.0, 6.0, 20001).reshape(1, -1)
        fp = W2 @ ((1.0 - np.tanh(W1 @ xs + b1) ** 2) * W1)
        sup = float(np.max(np.abs(fp)))
        bound = r['lipschitz_estimate']
        print(f"{tag:12s} true sup|f'| = {sup:.3f}  vs norm-product bound "
              f"{bound:.2f}  (certificate slack {bound / sup:.1f}x)")


if __name__ == '__main__':
    main()
