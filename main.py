"""
main.py
Entry point. Run with:

    python main.py                          # run all enabled experiments
    python main.py --group method_comparison # run only one group
    python main.py --name exp_method_ipopt   # run only one experiment
    python main.py --dry-run                 # print what would run, don't execute
    python main.py --no-plots                # skip figure generation (faster)

All configuration lives in configs/experiments.py.
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

from configs.experiments import EXPERIMENTS
from src.data import generate_dataset, generate_pendulum_dataset
from src.solver import solve
from src.logger import save_result, print_summary
from src.plotting import (plot_fit, plot_method_comparison, plot_lipschitz_sweep,
                           plot_size_scaling, plot_noise_robustness, plot_adam_convergence,
                           plot_multistart, plot_kkt_analysis, plot_penalty_vs_hard,
                           plot_warm_start, plot_constraint_geometry,
                           plot_convergence_rate, plot_hessian_comparison,
                           plot_complexity, plot_pendulum, plot_pendulum_sweep)
# New groups (multistart / kkt_analysis / penalty_vs_hard / constraint_geometry)
# need solver internals (dual variables) or a solver method (penalty_adam) that
# src/solver.py's unmodified solve() does not expose/know about -- see
# run_experiment() below. warm_start runs a whole tightening sweep at once and
# is handled specially in the main loop.
from src import (kkt, multistart, penalty_adam, warm_start, constraint_geometry,
                 convergence, gauss_newton)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def run_experiment(exp):
    # data_seed lets the 'multistart' group fix the training data while
    # varying only the init seed; every existing experiment has no
    # 'data_seed' key, so this falls back to exp['seed'] exactly as before.
    data_seed = exp.get('data_seed', exp['seed'])
    if exp.get('task') == 'pendulum':
        # physically meaningful task (Group 13): damped-pendulum system ID
        X_train, y_train, X_test, y_test = generate_pendulum_dataset(
            n_train=exp['n_train'], n_test=exp['n_test'],
            noise_std=exp['noise_std'], seed=data_seed, t_range=exp['x_range'],
        )
    else:
        X_train, y_train, X_test, y_test = generate_dataset(
            n_train=exp['n_train'], n_test=exp['n_test'], noise_std=exp['noise_std'],
            seed=data_seed, d_in=exp['d_in'], d_out=exp['d_out'],
            H_teacher=exp['H_teacher'], x_range=exp['x_range'],
        )
    if exp['method'] == 'penalty_adam':
        res = penalty_adam.solve_penalty_adam(exp, X_train, y_train, X_test, y_test)
    elif exp['method'] == 'gauss_newton':
        res = gauss_newton.solve_gauss_newton(exp, X_train, y_train, X_test, y_test)
    elif exp.get('group') == 'kkt_analysis':
        res = kkt.solve_with_dual(exp, X_train, y_train, X_test, y_test)
    elif exp.get('group') == 'constraint_geometry':
        res = constraint_geometry.solve_with_duals(exp, X_train, y_train, X_test, y_test)
    elif exp.get('group') == 'convergence_rate' and exp['method'] == 'ipopt':
        # needs IPOPT's per-iteration inf_pr/inf_du log; the adam run in this
        # group goes through the normal solve(), which already records
        # ||grad f||_inf per iteration as kkt_history.
        res = convergence.solve_with_iterates(exp, X_train, y_train, X_test, y_test)
    else:
        res = solve(exp, X_train, y_train, X_test, y_test)
    return res, (X_train, y_train, X_test, y_test)


def make_comparison_figures(all_results, args):
    def group(name):
        return [r for r in all_results.values() if r['group'] == name]

    g1 = group('method_comparison')
    if len(g1) > 1:
        plot_method_comparison(g1, os.path.join(FIGURES_DIR, 'fig_method_comparison.png'))

    g2 = group('lipschitz_sweep')
    if len(g2) > 1:
        plot_lipschitz_sweep(g2, os.path.join(FIGURES_DIR, 'fig_lipschitz_sweep.png'))

    g3 = group('size_scaling')
    if len(g3) > 1:
        plot_size_scaling(g3, os.path.join(FIGURES_DIR, 'fig_size_scaling.png'))

    g4 = group('noise_robustness')
    if len(g4) > 1:
        plot_noise_robustness(g4, os.path.join(FIGURES_DIR, 'fig_noise_robustness.png'))

    g5 = group('multistart')
    if len(g5) > 1:
        multistart.summarize_multistart(g5)
        plot_multistart(g5, os.path.join(FIGURES_DIR, 'fig_multistart.png'))

    g6 = group('kkt_analysis')
    if len(g6) > 1:
        plot_kkt_analysis(g6, os.path.join(FIGURES_DIR, 'fig_kkt_analysis.png'))

    g7 = group('penalty_vs_hard')
    if len(g7) > 1:
        plot_penalty_vs_hard(g7, os.path.join(FIGURES_DIR, 'fig_penalty_vs_hard.png'))

    g8 = group('warm_start')
    if len(g8) > 1:
        plot_warm_start(g8, os.path.join(FIGURES_DIR, 'fig_warm_start.png'))

    g9 = group('constraint_geometry')
    if len(g9) > 1:
        plot_constraint_geometry(g9, os.path.join(FIGURES_DIR, 'fig_constraint_geometry.png'))

    g10 = group('convergence_rate')
    if len(g10) > 1:
        plot_convergence_rate(g10, os.path.join(FIGURES_DIR, 'fig_convergence_rate.png'))

    g11 = group('hessian_comparison')
    if len(g11) > 1:
        plot_hessian_comparison(g11, os.path.join(FIGURES_DIR, 'fig_hessian_comparison.png'))

    # complexity race combines its own baselines with the size_scaling IPOPT
    # runs (identical sizes/data); fall back to results/ on disk when only
    # one of the two groups was run in this session.
    g12 = group('complexity')
    if len(g12) > 1:
        ipopt_sizes = group('size_scaling')
        if not ipopt_sizes:
            import json, glob
            ipopt_sizes = [json.load(open(f)) for f in
                           glob.glob(os.path.join(RESULTS_DIR, 'exp_size_H*.json'))]
        plot_complexity(g12 + ipopt_sizes, os.path.join(FIGURES_DIR, 'fig_complexity.png'))

    g13 = group('pendulum')
    if len(g13) > 1:
        X_train, y_train, _, _ = generate_pendulum_dataset(noise_std=0.05, seed=0)
        plot_pendulum(g13, X_train, y_train,
                      os.path.join(FIGURES_DIR, 'fig_pendulum.png'))

    g14 = group('pendulum_sweep')
    if len(g14) > 1:
        plot_pendulum_sweep(g14, os.path.join(FIGURES_DIR, 'fig_pendulum_sweep.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default=None, help='Only run experiments from this group')
    parser.add_argument('--name', default=None, help='Only run the experiment with this name')
    parser.add_argument('--dry-run', action='store_true', help='Print experiment list but do not run')
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    exps = [e for e in EXPERIMENTS if e['enabled']]
    if args.group:
        exps = [e for e in exps if e['group'] == args.group]
    if args.name:
        exps = [e for e in exps if e['name'] == args.name]

    print(f'\n{"=" * 70}')
    print('  Neural Network Weight Optimization as a Constrained NLP')
    print(f'{"=" * 70}')
    print(f'  Experiments to run: {len(exps)}')
    if args.group: print(f'  Group filter: {args.group}')
    if args.name:  print(f'  Name filter:  {args.name}')
    print()

    if args.dry_run:
        print('DRY RUN - experiments that would execute:')
        for e in exps:
            print(f'  [{e["group"]:20s}]  {e["name"]:28s}  method={e["method"]:6s}  H={e["H"]}')
        return

    all_results = {}
    t_wall_start = time.time()

    for i, exp in enumerate(exps, 1):
        print(f'[{i:2d}/{len(exps)}]  {exp["name"]}  (method={exp["method"]}) ... ',
              end='', flush=True)
        try:
            # warm_start is a single config that drives a whole cold-vs-warm
            # tightening sweep -> many result dicts, handled on its own.
            if exp.get('group') == 'warm_start':
                ws_results = warm_start.run_warm_start_study(exp)
                for r in ws_results:
                    save_result(r, RESULTS_DIR)
                    all_results[r['name']] = r
                print(f'OK  ({len(ws_results)} solves: cold + warm tightening sweep)')
                continue

            res, data = run_experiment(exp)
            status = 'OK' if res['success'] else 'FAILED'
            print(f'{status}  train_mse={res["train_mse"]:.5f}  '
                  f'test_mse={res["test_mse"]:.5f}  t={res["solve_time"]:.3f}s')

            save_result(res, RESULTS_DIR)
            all_results[exp['name']] = res

            if not args.no_plots:
                X_train, y_train, X_test, y_test = data
                plot_fit(res, X_train, y_train, X_test, y_test, exp['x_range'],
                          os.path.join(FIGURES_DIR, f'{exp["name"]}.png'))
                if exp['method'] == 'adam':
                    plot_adam_convergence(res, os.path.join(FIGURES_DIR, f'{exp["name"]}_convergence.png'))

        except Exception as e:
            print(f'ERROR: {e}')
            import traceback; traceback.print_exc()

    if not args.no_plots:
        print('\nGenerating comparison figures...')
        make_comparison_figures(all_results, args)

    print_summary(list(all_results.values()))

    total = time.time() - t_wall_start
    print(f'Total wall time: {total:.1f} s')
    print(f'Results   -> {RESULTS_DIR}/')
    print(f'Figures   -> {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
