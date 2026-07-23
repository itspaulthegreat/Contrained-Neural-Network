"""
src/multistart.py
Multi-start / local-minima analysis.

The Lipschitz-constrained NLP is nonconvex (the constraint is bilinear
in the layer norms, and the tanh network itself is already nonconvex),
so IPOPT's converged point can depend on the initial guess. The
'multistart' experiment group (see configs/experiments.py) already
produces 20 ordinary IPOPT solves -- same training data, same L_max,
only the random initialization seed differs -- through the existing,
unmodified solve() pipeline. This module adds no new solving logic; it
only summarizes and reports on results that pipeline already produced.
"""

import numpy as np


def summarize_multistart(results):
    """
    results: list of result dicts from the 'multistart' group (same data,
             same L_max, 20 different random init seeds).

    Prints the best / median / worst run by final training MSE and a
    rough count of distinct objective values found, then returns a small
    summary dict.
    """
    if not results:
        return {}

    by_mse = sorted(results, key=lambda r: r['train_mse'])
    best, worst = by_mse[0], by_mse[len(by_mse) - 1]
    median = by_mse[len(by_mse) // 2]

    mses = np.array([r['train_mse'] for r in results])
    n_distinct = len(np.unique(np.round(mses, 5)))

    print(f'\nMulti-start study over {len(results)} random initializations '
          f'(same data, same L_max):')
    print(f'  best   train MSE = {best["train_mse"]:.6f}  '
          f'(lipschitz~{best["lipschitz_estimate"]:.3f}, {best["name"]})')
    print(f'  median train MSE = {median["train_mse"]:.6f}  '
          f'(lipschitz~{median["lipschitz_estimate"]:.3f}, {median["name"]})')
    print(f'  worst  train MSE = {worst["train_mse"]:.6f}  '
          f'(lipschitz~{worst["lipschitz_estimate"]:.3f}, {worst["name"]})')
    print(f'  spread (worst - best) = {worst["train_mse"] - best["train_mse"]:.6f}')
    print(f'  distinct objective values (5-decimal rounding): '
          f'{n_distinct} out of {len(results)} runs\n')

    return dict(
        best=best, median=median, worst=worst,
        mean_mse=float(mses.mean()), std_mse=float(mses.std()),
        n_distinct=n_distinct,
    )
