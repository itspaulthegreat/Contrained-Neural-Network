"""Multi-start / local-minima analysis. The constrained NLP is nonconvex, so
IPOPT's converged point can depend on the initial guess. This module summarizes
the 'multistart' group's solves (same data and L_max, different init seeds)."""

import numpy as np


def summarize_multistart(results):
    """Report best/median/worst training MSE and count of distinct optima."""
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
