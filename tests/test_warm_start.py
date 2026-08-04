"""Tests the warm-start claim: using the solution of a looser Lipschitz bound
as the initial guess for a tighter one is cheaper (fewer IPOPT iterations) and
reaches an equivalent-quality solution that still respects the constraint."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.experiments import _make
from src.data import generate_dataset
from src.warm_start import solve_ipopt_from


def _base_cfg(L_max):
    cfg = _make('test_warm', 'warm-start unit test', 'warm_start',
                method='ipopt', H=6, L_max=L_max,
                use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
                noise_std=0.1, seed=0)
    return cfg


def _data():
    return generate_dataset(n_train=60, n_test=40, noise_std=0.1, seed=0,
                            d_in=1, d_out=1, H_teacher=6, x_range=(-3.0, 3.0))


def _solve(L_max, x0=None):
    cfg = _base_cfg(L_max)
    X_train, y_train, X_test, y_test = _data()
    return solve_ipopt_from(cfg, X_train, y_train, X_test, y_test, x0=x0)


# Solve once at the tighter bound from scratch (cold) ...
_cold = _solve(4.0)
# ... and once from the looser (L_max=8.0) solution as the warm start.
_loose = _solve(8.0)
_warm = _solve(4.0, x0=_loose['w'])


def test_warm_start_uses_fewer_iterations():
    """Warm-started solve takes fewer IPOPT iterations than the cold solve."""
    assert _warm['n_iter'] < _cold['n_iter'], (
        f"warm n_iter={_warm['n_iter']} not fewer than cold n_iter={_cold['n_iter']}")


def test_warm_start_respects_lipschitz_constraint():
    """The warm-started solution still satisfies the tighter bound."""
    assert _warm['max_constraint_violation'] <= 1e-9


def test_warm_and_cold_reach_similar_quality():
    """Warm and cold solves reach essentially the same training error."""
    rel = abs(_warm['train_mse'] - _cold['train_mse']) / _cold['train_mse']
    assert rel < 0.10, f"train_mse differ by {rel:.1%} (warm={_warm['train_mse']}, cold={_cold['train_mse']})"
