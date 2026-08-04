"""
Noisy data generation for the synthetic teacher-student task.

Provides the three-way (train / validation / test) split used by the reported
synthetic experiment (``experiments/exact_penalty.py``). The Lipschitz bound L
and norm-ball bound B are fixed requirements set in the experiment itself, not
selected here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import make_teacher, _teacher_forward

SEED = 2
SIGMA = 0.2
H = 8
X_RANGE = (-3.0, 3.0)


def three_way_split(seed=SEED, n=40, sigma=SIGMA):
    """Independent train / validation / test draws from the same teacher."""
    rng = np.random.default_rng(seed)
    teacher = make_teacher(seed=seed)
    out = []
    for _ in range(3):
        X = rng.uniform(X_RANGE[0], X_RANGE[1], size=(1, n))
        y = _teacher_forward(X, *teacher) + rng.normal(0, sigma, size=(1, n))
        out.append((X, y))
    return out
