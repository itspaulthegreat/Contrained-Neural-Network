"""
Ablation of the symmetry-breaking (ordered-bias) constraint.

The hidden units of a one-layer network can be permuted without changing the
function, so every minimiser has H! relabelled copies. The ordered-bias
constraint b1[0] <= ... <= b1[H-1] picks one canonical copy. This study asks
whether that constraint is doing anything useful: for a range of random
initialisations, each problem is solved twice -- with and without the ordering
constraint, all else identical (Lipschitz and norm-ball active in both) -- and
we compare fit quality, solve cost, constraint activity, and whether the two
solutions are permutations of one another.

    python symmetry_ablation.py
        -> results/symmetry_ablation.json
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.experiments import _make
from src.data import generate_dataset, generate_pendulum_dataset
from src.model import param_shapes, unflatten_numpy
from src.solver import solve

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(8))
H = 8
GAP_TOL = 1e-4          # a consecutive-bias gap below this counts the order constraint as active


def canonical(w, shapes):
    """Sort hidden units by ascending b1 and return the reordered weight vector."""
    W1, b1, W2, b2 = unflatten_numpy(w, shapes)
    order = np.argsort(b1.ravel())
    return np.concatenate([W1[order].ravel(order='F'), b1[order].ravel(order='F'),
                           W2[:, order].ravel(order='F'), b2.ravel(order='F')])


def min_adjacent_gap(w, shapes):
    """Smallest gap between consecutive (sorted) hidden biases. Near zero means
    two units share a bias -- i.e. the ordering constraint is binding there."""
    _, b1, _, _ = unflatten_numpy(w, shapes)
    return float(np.min(np.diff(np.sort(b1.ravel()))))


def already_ordered(w, shapes, tol=1e-6):
    """True if the biases came out ascending without being told to."""
    _, b1, _, _ = unflatten_numpy(w, shapes)
    return bool(np.all(np.diff(b1.ravel()) >= -tol))


def run_task(task_name, data):
    X_train, y_train, X_test, y_test = data
    shapes = param_shapes(1, H, 1)
    rows = []
    for seed in SEEDS:
        common = dict(H=H, use_lipschitz=True, use_norm_ball=True,
                      L_max=4.0, B_max=6.0, noise_std=0.05, seed=seed)
        exp_on = _make(f'sym_on_{task_name}_s{seed}', 'ordered', 'symmetry_ablation',
                       method='ipopt', use_symmetry_break=True, **common)
        exp_off = _make(f'sym_off_{task_name}_s{seed}', 'free', 'symmetry_ablation',
                        method='ipopt', use_symmetry_break=False, **common)
        r_on = solve(exp_on, X_train, y_train, X_test, y_test)
        r_off = solve(exp_off, X_train, y_train, X_test, y_test)

        w_on = np.asarray(r_on['w']); w_off = np.asarray(r_off['w'])
        perm_gap = float(np.linalg.norm(canonical(w_on, shapes) - canonical(w_off, shapes)))
        rows.append(dict(
            seed=seed,
            test_on=r_on['test_mse'], test_off=r_off['test_mse'],
            train_on=r_on['train_mse'], train_off=r_off['train_mse'],
            iter_on=r_on['n_iter'], iter_off=r_off['n_iter'],
            time_on=r_on['solve_time'], time_off=r_off['solve_time'],
            success_on=r_on['success'], success_off=r_off['success'],
            viol_on=r_on['max_constraint_violation'], viol_off=r_off['max_constraint_violation'],
            min_gap_on=min_adjacent_gap(w_on, shapes),
            order_active_on=min_adjacent_gap(w_on, shapes) < GAP_TOL,   # constraint binding at the optimum
            free_already_ordered=already_ordered(w_off, shapes),        # did order emerge on its own?
            same_minimiser=perm_gap < 1e-3,                             # identical up to a hidden-unit permutation
            perm_distance=perm_gap,
        ))
    return rows


def summarise(name, rows):
    d_test = np.array([r['test_on'] - r['test_off'] for r in rows])   # + means ordering HURT the fit
    active_on = sum(r['order_active_on'] for r in rows)
    free_ordered = sum(r['free_already_ordered'] for r in rows)
    worse_when_active = sum(r['order_active_on'] and r['test_on'] > r['test_off'] + 1e-5 for r in rows)
    all_feasible = all(r['viol_on'] < 1e-6 and r['viol_off'] < 1e-6
                       and r['success_on'] and r['success_off'] for r in rows)
    print(f"\n[{name}]  (H={H}, {len(rows)} random inits, Lipschitz+ball on in both)")
    print(f"  {'seed':>4} {'test_on':>9} {'test_off':>9} {'d_test':>9} {'it_on':>6} {'it_off':>6} "
          f"{'gap_on':>8} {'active':>7}")
    for r in rows:
        print(f"  {r['seed']:>4} {r['test_on']:>9.5f} {r['test_off']:>9.5f} "
              f"{r['test_on']-r['test_off']:>+9.5f} {r['iter_on']:>6} {r['iter_off']:>6} "
              f"{r['min_gap_on']:>8.1e} {str(r['order_active_on']):>7}")
    print(f"  all runs feasible and converged: {all_feasible}")
    print(f"  mean (test_on - test_off) = {np.mean(d_test):+.2e}  (+ = ordering degrades the fit)")
    print(f"  ordering constraint active (binding) at the optimum: {active_on}/{len(rows)}")
    print(f"  runs where active AND fit is worse than free: {worse_when_active}/{len(rows)}")
    print(f"  free solve's biases emerge already ordered on their own: {free_ordered}/{len(rows)}")


def main():
    tasks = {
        'synthetic': generate_dataset(noise_std=0.05, seed=0),
        'pendulum': generate_pendulum_dataset(noise_std=0.05, seed=0, t_range=(0.0, 6.0)),
    }
    out = {}
    for name, data in tasks.items():
        rows = run_task(name, data)
        summarise(name, rows)
        out[name] = rows
    path = os.path.join(HERE, 'results', 'symmetry_ablation.json')
    json.dump(dict(H=H, seeds=SEEDS, tasks=out), open(path, 'w'), indent=1, default=float)
    print('\nwrote', path)


if __name__ == '__main__':
    main()
