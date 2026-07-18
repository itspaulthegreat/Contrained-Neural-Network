# -*- coding: utf-8 -*-
"""Point 12 check: is the stage-4 (symmetry) solution of surface_ablation.py a
re-indexed copy of stage 3, or a genuinely different point? Prints b1 vectors,
the best hidden-unit permutation match, and distances in weight space and in
the 2-D slice."""
import sys, itertools
import numpy as np

sys.path.insert(0, r"C:\Users\arind\Desktop\project nop\nn_constrained_nlp")
from src.data import generate_dataset
from src.model import param_shapes, mse_numpy, unflatten_numpy
from surface_ablation import solve, SEED, SIGMA, H

Xtr, ytr, _, _ = generate_dataset(n_train=40, n_test=40, noise_std=SIGMA, seed=SEED)
shapes = param_shapes(1, H, 1)

w2 = solve(Xtr, ytr, use_lip=True)
B = round(float(np.linalg.norm(w2)) * 0.8, 2)
w3 = solve(Xtr, ytr, use_lip=True, use_ball=True, B=B)
w4 = solve(Xtr, ytr, use_lip=True, use_ball=True, B=B, use_sym=True)

W1a, b1a, W2a, b2a = unflatten_numpy(w3, shapes)
W1b, b1b, W2b, b2b = unflatten_numpy(w4, shapes)

np.set_printoptions(precision=4, suppress=True)
print("stage-3 b1 (unordered):", b1a.flatten())
print("stage-4 b1:            ", b1b.flatten())
print("stage-4 b1 sorted?     ", bool(np.all(np.diff(b1b.flatten()) >= -1e-9)))
print("stage-3 b1 sorted?     ", bool(np.all(np.diff(b1a.flatten()) >= -1e-9)))
print("f(w3) =", mse_numpy(w3, Xtr, ytr, shapes), "  f(w4) =", mse_numpy(w4, Xtr, ytr, shapes))
print("||w4 - w3|| =", np.linalg.norm(w4 - w3))

# best permutation of stage-3 hidden units matched to stage-4
# (a hidden unit = the triple (W1 row, b1 entry, W2 column); tanh is odd, so a
#  sign flip of a unit's (W1,b1) with its W2 entry is also an equivalence)
units_a = [(W1a[i, 0], b1a[i, 0] if b1a.ndim > 1 else b1a[i], W2a[0, i]) for i in range(H)]
units_b = [(W1b[i, 0], b1b[i, 0] if b1b.ndim > 1 else b1b[i], W2b[0, i]) for i in range(H)]
ua = np.array(units_a); ub = np.array(units_b)

# greedy match each stage-4 unit to nearest stage-3 unit (allowing sign flip)
used = set(); rows = []
for j in range(H):
    best = (None, np.inf, +1)
    for i in range(H):
        if i in used:
            continue
        d_plus = np.linalg.norm(ub[j] - ua[i])
        d_minus = np.linalg.norm(ub[j] - np.array([-ua[i][0], -ua[i][1], -ua[i][2]]))
        if d_plus < best[1]:
            best = (i, d_plus, +1)
        if d_minus < best[1]:
            best = (i, d_minus, -1)
    used.add(best[0]); rows.append((j, best[0], best[1], best[2]))
print("\nstage-4 unit <- stage-3 unit   distance   sign")
for j, i, d, s in rows:
    print(f"      {j}     <-      {i}        {d:9.5f}    {'+' if s > 0 else '-'}")
worst = max(r[2] for r in rows)
print(f"\nworst matched-unit distance: {worst:.5f}")
print("=> stage 4 is a re-indexed copy of stage 3" if worst < 1e-2 else
      "=> stage 4 is NOT simply a permuted copy — it is a nearby different KKT point")
