"""
src/plotting.py
──────────────────
All figures. Each function takes result dict(s) and a save path, and
writes a PNG. No figure is ever shown interactively (Agg backend) so
this runs fine from the command line / over SSH.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.model import param_shapes, forward_numpy

# ── Publication-quality defaults, applied to every figure in this module ─────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Consistent solver color palette used across all comparison plots.
METHOD_COLORS = {
    'ipopt': 'tab:blue',
    'sqp': 'tab:orange',
    'adam': 'tab:green',
    'penalty_adam': 'tab:red',
}


# ─────────────────────────────────────────────────────────────────────────
#  Single-experiment fit plot (data scatter + learned curve)
# ─────────────────────────────────────────────────────────────────────────

def plot_fit(res, X_train, y_train, X_test, y_test, x_range, path):
    shapes = param_shapes(1, res['H'], 1)
    w = np.array(res['w'])

    xs = np.linspace(x_range[0], x_range[1], 300).reshape(1, -1)
    ys = forward_numpy(w, xs, shapes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train.flatten(), y_train.flatten(), s=18, c='tab:blue',
               alpha=0.7, label='train data')
    ax.scatter(X_test.flatten(), y_test.flatten(), s=18, c='tab:orange',
               alpha=0.7, marker='^', label='test data')
    ax.plot(xs.flatten(), ys.flatten(), c='black', lw=2, label='learned f(x; w)')
    ax.set_title(f"train MSE={res['train_mse']:.4f}   test MSE={res['test_mse']:.4f}")
    ax.set_xlabel('input $x$ [-]'); ax.set_ylabel('output $y$ [-]')
    ax.legend(loc='best', fontsize=8)
    fig.suptitle(res['label'])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Method comparison: IPOPT vs SQP vs Adam
# ─────────────────────────────────────────────────────────────────────────

def _method_label(r):
    """Bar label that distinguishes the exact-spectral IPOPT run from the
    Frobenius-Lipschitz IPOPT run (both have method='ipopt')."""
    if r.get('use_spectral_norm'):
        return 'IPOPT\n(spectral)'
    if r['method'] == 'ipopt':
        return 'IPOPT\n(Frobenius)'
    return r['method'].upper()


def plot_method_comparison(results, path):
    results = sorted(results, key=lambda r: (r['method'], r.get('use_spectral_norm', False)))
    names = [_method_label(r) for r in results]
    colors = [METHOD_COLORS.get(r['method'], 'tab:gray') for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    solve_time = [r['solve_time'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(names))
    w = 0.35

    axes[0].bar(x - w / 2, train_mse, w, label='train MSE', color='tab:blue')
    axes[0].bar(x + w / 2, test_mse, w, label='test MSE', color='tab:orange')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel('MSE [-]'); axes[0].set_title('Fit quality')
    axes[0].legend(fontsize=8)

    axes[1].bar(x, solve_time, color=colors)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel('Solve time [s]'); axes[1].set_title('Computational cost')
    axes[1].set_yscale('log')

    fig.suptitle('Optimizer comparison — same network, same data, same constraints')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Lipschitz bound sweep: train/test MSE trade-off
# ─────────────────────────────────────────────────────────────────────────

def plot_lipschitz_sweep(results, path):
    results = sorted(results, key=lambda r: r['L_max'])
    L = [r['L_max'] for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    lip_achieved = [r['lipschitz_estimate'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(L, train_mse, 'o-', color='tab:blue', label='train MSE')
    axes[0].plot(L, test_mse, 's-', color='tab:orange', label='test MSE')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Lipschitz bound $L_{max}$ [-]')
    axes[0].set_ylabel('MSE [-]')
    axes[0].set_title('Fit vs. generalization trade-off')
    axes[0].legend(fontsize=8)

    axes[1].plot(L, lip_achieved, 'd-', color='tab:purple', label='achieved $\\|W_1\\|_F\\|W_2\\|_F$')
    axes[1].plot(L, L, '--', color='gray', label='bound $L_{max}$')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('Lipschitz bound $L_{max}$ [-]')
    axes[1].set_ylabel('achieved Lipschitz estimate [-]')
    axes[1].set_title('Constraint activity')
    axes[1].legend(fontsize=8)

    fig.suptitle('Effect of tightening the Lipschitz constraint (IPOPT)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  NLP size scaling: solve time / iterations vs number of decision vars
# ─────────────────────────────────────────────────────────────────────────

def plot_size_scaling(results, path):
    results = sorted(results, key=lambda r: r['n_vars'])
    n_vars = [r['n_vars'] for r in results]
    solve_time = [r['solve_time'] for r in results]
    n_iter = [r['n_iter'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(n_vars, solve_time, 'o-', color='tab:blue')
    axes[0].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[0].set_ylabel('Solve time [s]')
    axes[0].set_title('Solve time vs. problem size')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')

    axes[1].plot(n_vars, n_iter, 's-', color='tab:blue')
    axes[1].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[1].set_ylabel('IPOPT iterations [-]')
    axes[1].set_title('Iterations vs. problem size')
    axes[1].set_xscale('log')

    fig.suptitle('NLP scaling with network size (hidden units H)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Noise robustness: constrained (IPOPT) vs unconstrained (Adam)
# ─────────────────────────────────────────────────────────────────────────

def plot_noise_robustness(results, path):
    noise_levels = sorted(set(r['noise_std'] for r in results))
    methods = ['ipopt', 'adam']
    colors = {'ipopt': METHOD_COLORS['ipopt'], 'adam': METHOD_COLORS['adam']}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(noise_levels))
    w = 0.35

    for i, method in enumerate(methods):
        vals = []
        for nl in noise_levels:
            match = [r for r in results if r['method'] == method and r['noise_std'] == nl]
            vals.append(match[0]['test_mse'] if match else np.nan)
        offset = (i - 0.5) * w
        ax.bar(x + offset, vals, w, label=method.upper(), color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={nl}' for nl in noise_levels])
    ax.set_xlabel('Label noise std $\\sigma$ [-]')
    ax.set_ylabel('Test MSE [-]')
    ax.set_title('constrained (IPOPT) vs unconstrained (Adam)')
    ax.legend(fontsize=8)
    fig.suptitle('Generalization under noisy data')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Adam convergence curve
# ─────────────────────────────────────────────────────────────────────────

def plot_adam_convergence(res, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res['history'], color=METHOD_COLORS['adam'])
    ax.set_yscale('log')
    ax.set_xlabel('Iteration [-]')
    ax.set_ylabel('Training MSE [-] (log scale)')
    ax.set_title(f"Adam convergence — {res['name']}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Multi-start / local minima study
# ─────────────────────────────────────────────────────────────────────────

def plot_multistart(results, path):
    train_mse = np.array([r['train_mse'] for r in results])
    lip = np.array([r['lipschitz_estimate'] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].hist(train_mse, bins=min(10, len(results)), color='tab:blue', edgecolor='black')
    axes[0].set_xlabel('Final train MSE [-]')
    axes[0].set_ylabel('Count [-]')
    axes[0].set_title(f'Spread of local minima ({len(results)} random starts)')

    axes[1].scatter(train_mse, lip, color='tab:blue')
    axes[1].set_xlabel('Final train MSE [-]')
    axes[1].set_ylabel('Lipschitz estimate $\\|W_1\\|_F\\|W_2\\|_F$ [-]')
    axes[1].set_title('Objective vs. achieved Lipschitz value')

    fig.suptitle('Multi-start study — IPOPT from 20 random initializations\n'
                  '(same data, same $L_{max}$, only init seed differs)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  KKT / dual-variable analysis
# ─────────────────────────────────────────────────────────────────────────

def plot_kkt_analysis(results, path):
    results = sorted(results, key=lambda r: r['L_max'])
    L = [r['L_max'] for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    lam = [r['lam_lipschitz'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(L, train_mse, 'o-', color='tab:blue', label='train MSE')
    axes[0].plot(L, test_mse, 's-', color='tab:orange', label='test MSE')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Lipschitz bound $L_{max}$ [-]')
    axes[0].set_ylabel('MSE [-]')
    axes[0].set_title('Fit vs. generalization (for reference)')
    axes[0].legend(fontsize=8)

    axes[1].plot(L, lam, 'o-', color='tab:purple')
    axes[1].axhline(0.0, color='gray', linestyle='--', lw=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Lipschitz bound $L_{max}$ [-]')
    axes[1].set_ylabel('Dual variable $\\lambda$ [-] (Lipschitz)')
    axes[1].set_title('Shadow price: active ($\\lambda>0$) vs. slack ($\\lambda\\approx0$)')

    fig.suptitle('KKT dual-variable analysis of the Lipschitz constraint (IPOPT)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Penalty method (Adam) vs. hard constraint (IPOPT)
# ─────────────────────────────────────────────────────────────────────────

def plot_penalty_vs_hard(results, path):
    ipopt_ref = [r for r in results if r['method'] == 'ipopt']
    penalty = sorted([r for r in results if r['method'] == 'penalty_adam'],
                      key=lambda r: r['rho'])

    # rho=0 can't be placed on a log axis -- plot it at a small placeholder
    # position instead (annotated below) so the rest of the sweep still
    # reads as a clean log scale instead of symlog's mirrored negative ticks.
    rho_eps = 3e-6
    rho = [max(r['rho'], rho_eps) for r in penalty]
    violation = [r['max_constraint_violation'] for r in penalty]
    test_mse = [r['test_mse'] for r in penalty]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    c_pen, c_ipopt = METHOD_COLORS['penalty_adam'], METHOD_COLORS['ipopt']

    axes[0].plot(rho, violation, 'o-', color=c_pen, label='penalty Adam')
    if ipopt_ref:
        axes[0].axhline(ipopt_ref[0]['max_constraint_violation'], color=c_ipopt,
                         linestyle='--', label='IPOPT (hard constraint)')
    axes[0].annotate('$\\rho=0$', xy=(rho[0], violation[0]), xytext=(10, -12),
                      textcoords='offset points', fontsize=7, ha='left')
    axes[0].set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=1e-6)
    axes[0].set_xlabel('Penalty weight $\\rho$ [-]')
    axes[0].set_ylabel('Lipschitz constraint violation [-]')
    axes[0].set_title('Constraint violation: penalty vs. exact')
    axes[0].legend(fontsize=8)

    axes[1].plot(rho, test_mse, 's-', color=c_pen, label='penalty Adam')
    if ipopt_ref:
        axes[1].axhline(ipopt_ref[0]['test_mse'], color=c_ipopt,
                         linestyle='--', label='IPOPT (hard constraint)')
    axes[1].annotate('$\\rho=0$', xy=(rho[0], test_mse[0]), xytext=(0, 8),
                      textcoords='offset points', fontsize=7, ha='center')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Penalty weight $\\rho$ [-]')
    axes[1].set_ylabel('Test MSE [-]')
    axes[1].set_title('Fit quality: penalty vs. exact')
    axes[1].legend(fontsize=8)

    fig.suptitle('Penalty method (Adam) vs. hard constraint (IPOPT)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Warm-start study (GROUP 8): cold vs. warm over a tightening L_max sweep
# ─────────────────────────────────────────────────────────────────────────

def plot_warm_start(results, path):
    cold = sorted([r for r in results if r['strategy'] == 'cold'], key=lambda r: r['L_max'])
    warm = sorted([r for r in results if r['strategy'] == 'warm'], key=lambda r: r['L_max'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    series = [
        (cold, 'cold start (same $w_0$)', 'tab:gray', 'o'),
        (warm, 'warm start (previous solution)', 'tab:blue', 's'),
    ]
    for data, label, color, marker in series:
        L = [r['L_max'] for r in data]
        n_iter = [r['n_iter'] for r in data]
        axes[0].plot(L, n_iter, marker + '-', color=color, label=label)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Lipschitz bound $L_{max}$ [-] (tightening $\\rightarrow$)')
    axes[0].set_ylabel('IPOPT iterations [-]')
    axes[0].set_title('Iteration count vs. $L_{max}$')
    axes[0].legend(fontsize=8)

    for data, label, color, marker in series:
        L = [r['L_max'] for r in data]
        t = [r['solve_time'] for r in data]
        axes[1].plot(L, t, marker + '-', color=color, label=label)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Lipschitz bound $L_{max}$ [-] (tightening $\\rightarrow$)')
    axes[1].set_ylabel('Solve time [s]')
    axes[1].set_title('Solve time vs. $L_{max}$')
    axes[1].legend(fontsize=8)

    fig.suptitle('Warm-start study — incremental Lipschitz tightening (IPOPT)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Constraint geometry (GROUP 9): which constraint binds as B_max varies
# ─────────────────────────────────────────────────────────────────────────

def plot_constraint_geometry(results, path):
    results = sorted(results, key=lambda r: r['B_max'])
    B = [r['B_max'] for r in results]
    train_mse = [r['train_mse'] for r in results]
    test_mse = [r['test_mse'] for r in results]
    lam_lip = [r['lam_lipschitz'] for r in results]
    lam_nb = [r['lam_norm_ball'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(B, train_mse, 'o-', color='tab:blue', label='train MSE')
    axes[0].plot(B, test_mse, 's-', color='tab:orange', label='test MSE')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Norm-ball radius $B_{max}$ [-]')
    axes[0].set_ylabel('MSE [-]')
    axes[0].set_title('Fit vs. norm-ball radius')
    axes[0].legend(fontsize=8)

    axes[1].plot(B, lam_lip, 'o-', color='tab:purple', label='$\\lambda$ Lipschitz')
    axes[1].plot(B, lam_nb, 'd-', color='tab:red', label='$\\lambda$ norm-ball')
    axes[1].axhline(0.0, color='gray', linestyle='--', lw=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Norm-ball radius $B_{max}$ [-]')
    axes[1].set_ylabel('Dual variable $\\lambda$ [-]')
    axes[1].set_title('Which constraint binds (active $\\Leftrightarrow \\lambda>0$)')
    axes[1].legend(fontsize=8)

    fig.suptitle('Constraint geometry — Lipschitz vs. norm-ball interaction (IPOPT)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Complexity race (GROUP 12 + GROUP 3): all solvers vs growing NLP size
# ─────────────────────────────────────────────────────────────────────────

def plot_complexity(results, path):
    """results: the 'complexity' group (adam + gauss_newton per H) PLUS the
    'size_scaling' group (constrained IPOPT per H) — same data and sizes."""
    series = [
        ('ipopt', 'IPOPT (constrained, exact Hessian)', METHOD_COLORS['ipopt'], 'o'),
        ('gauss_newton', 'Gauss-Newton/LM (unconstrained)', 'tab:red', 'd'),
        ('adam', 'Adam (unconstrained)', METHOD_COLORS['adam'], 's'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for method, label, color, marker in series:
        data = sorted([r for r in results if r['method'] == method],
                      key=lambda r: r['n_vars'])
        if not data:
            continue
        n_vars = [r['n_vars'] for r in data]
        axes[0].plot(n_vars, [r['solve_time'] for r in data],
                     marker + '-', color=color, label=label)
        # test MSE, not train: unconstrained methods drive train MSE down as
        # capacity grows (overfitting) -- generalization is the honest metric.
        axes[1].plot(n_vars, [r['test_mse'] for r in data],
                     marker + '-', color=color, label=label)

    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[0].set_ylabel('Solve time [s]')
    axes[0].set_title('Cost vs. size')
    axes[0].legend(fontsize=8)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[1].set_ylabel('Test MSE [-]')
    axes[1].set_title('Generalization vs. size')
    axes[1].legend(fontsize=8)

    fig.suptitle('Complexity race — same data, growing network (H = 4 → 64)\n'
                  '(IPOPT solves the constrained problem; Adam and GN/LM the unconstrained one)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Pendulum system ID (GROUP 13): physical task, physical units
# ─────────────────────────────────────────────────────────────────────────

def plot_pendulum(results, X_train, y_train, path):
    from src.data import pendulum_true
    ts = np.linspace(0.0, 6.0, 400)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, pendulum_true(ts), color='black', lw=2, ls=':',
            label='true response $\\theta(t)$ (physics)')
    ax.scatter(X_train.flatten(), y_train.flatten(), s=18, c='tab:blue',
               alpha=0.6, label='noisy measurements')

    styles = {
        'exp_pendulum_ipopt': ('IPOPT, $L_{max}=4$ rad/s (certified)', 'tab:purple', '-'),
        'exp_pendulum_tight': ('IPOPT, $L_{max}=1$ rad/s (over-tight)', 'tab:red', '--'),
        'exp_pendulum_adam': ('Adam (unconstrained)', METHOD_COLORS['adam'], '-'),
    }
    for r in sorted(results, key=lambda r: r['name']):
        label, color, ls = styles.get(r['name'], (r['name'], 'gray', '-'))
        shapes = param_shapes(1, r['H'], 1)
        ys = forward_numpy(np.array(r['w']), ts.reshape(1, -1), shapes)
        ax.plot(ts, ys.flatten(), color=color, ls=ls, lw=2,
                label=f"{label} — test MSE {r['test_mse']:.4f}")

    ax.set_xlabel('time $t$ [s]')
    ax.set_ylabel('angle $\\theta$ [rad]')
    ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Physical task — damped-pendulum system identification\n'
                  'the Lipschitz bound caps the model\'s angular rate '
                  '$|\\mathrm{d}\\hat{\\theta}/\\mathrm{d}t| \\leq L_{max}$')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Convergence rate (GROUP 10): KKT residual vs. iteration, IPOPT vs. Adam
# ─────────────────────────────────────────────────────────────────────────

def _conv_style(r):
    """(label, color, linestyle) for one convergence_rate run."""
    if r['method'] == 'adam':
        return 'Adam (unconstrained, $\\|\\nabla f\\|_\\infty$)', METHOD_COLORS['adam'], '-'
    if r['method'] == 'gauss_newton':
        return 'Gauss-Newton/LM (unconstrained, self-implemented)', 'tab:red', '-.'
    if r.get('use_lipschitz'):
        return 'IPOPT (Lipschitz-constrained)', 'tab:purple', '--'
    return 'IPOPT (unconstrained)', METHOD_COLORS['ipopt'], '-'


def plot_convergence_rate(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ipopt_max_iter = max((len(r.get('kkt_history') or [])
                          for r in results if r['method'] == 'ipopt'), default=0)

    for r in sorted(results, key=lambda r: r['method']):
        h = np.asarray(r.get('kkt_history') or [], dtype=float)
        if len(h) == 0:
            continue
        label, color, ls = _conv_style(r)
        it = np.arange(len(h))
        # log axes -- clip exact zeros to the smallest positive value seen
        h = np.maximum(h, np.min(h[h > 0]) if np.any(h > 0) else 1e-16)
        # Running minimum: IPOPT's raw per-iteration residual oscillates
        # (barrier-parameter updates, rejected line-search steps), so the
        # monotone envelope "best residual reached so far" is what makes the
        # convergence *rate* readable.
        h = np.minimum.accumulate(h)
        axes[0].plot(it, h, ls, color=color, label=label)
        axes[1].plot(it, h, ls, color=color, label=label)

    axes[0].set_xscale('symlog', linthresh=1)  # iteration 0 has no log position
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration $k$ [-]')
    axes[0].set_ylabel('best KKT residual so far [-]')
    axes[0].set_title('Full run (log-log)')
    axes[0].legend(fontsize=8)

    axes[1].set_yscale('log')
    axes[1].set_xlim(0, max(ipopt_max_iter, 1) * 1.05)
    axes[1].set_xlabel('Iteration $k$ [-]')
    axes[1].set_ylabel('best KKT residual so far [-]')
    axes[1].set_title('Zoom on IPOPT range: superlinear tail')
    axes[1].legend(fontsize=8)

    fig.suptitle('Convergence rate — KKT residual per iteration\n'
                  '(same objective, same data, same initial guess $w_0$)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Hessian comparison (GROUP 11): exact vs. L-BFGS Hessian in IPOPT
# ─────────────────────────────────────────────────────────────────────────

def plot_hessian_comparison(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    series = [
        ('exact', 'exact Hessian (AD)', 'tab:blue', 'o'),
        ('limited-memory', 'L-BFGS approximation', 'tab:red', 's'),
    ]
    for mode, label, color, marker in series:
        data = sorted([r for r in results if r.get('hessian_mode') == mode],
                      key=lambda r: r['n_vars'])
        if not data:
            continue
        n_vars = [r['n_vars'] for r in data]
        axes[0].plot(n_vars, [r['solve_time'] for r in data],
                     marker + '-', color=color, label=label)
        axes[1].plot(n_vars, [r['n_iter'] for r in data],
                     marker + '-', color=color, label=label)

    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[0].set_ylabel('Solve time [s]')
    axes[0].set_title('Total solve time')
    axes[0].legend(fontsize=8)

    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('Number of decision variables (NLP size) [-]')
    axes[1].set_ylabel('IPOPT iterations [-]')
    axes[1].set_title('Iteration count')
    axes[1].legend(fontsize=8)

    fig.suptitle('IPOPT: exact vs. limited-memory Hessian (same constrained NLP)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
