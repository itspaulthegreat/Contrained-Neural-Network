"""
Noisy data generation for the damped-pendulum task.

Provides the three-way (train / validation / test) split of the pendulum's
noisy free response used by the reported pendulum experiment
(``experiments/pendulum_convergence.py``) and by ``experiments/report_figures.py``.
The Lipschitz bound L and norm-ball bound B are fixed requirements set in the
experiment itself, not selected here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import pendulum_true

SEED = 0
SIGMA = 0.15
H = 8
T_RANGE = (0.0, 6.0)


def three_way_split_pendulum(seed=SEED, n=60, sigma=SIGMA):
    """Three independent draws of the pendulum's noisy response."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        T = rng.uniform(T_RANGE[0], T_RANGE[1], size=(1, n))
        y = pendulum_true(T) + rng.normal(0, sigma, size=(1, n))
        out.append((T, y))
    return out
