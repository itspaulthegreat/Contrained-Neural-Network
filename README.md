# Neural Network Weight Optimization as a Constrained NLP

## What this project actually is (plain English)

Normally, people train a neural network with gradient descent (Adam) and add
regularization as a *penalty* in the loss. Here, instead, we treat finding
the network's weights as a **Nonlinear Program (NLP)** and solve it with a
real NLP solver (**IPOPT**, interior-point; and **SQP**), and we enforce
regularization as **hard constraints** rather than penalties.

The neural network is not the point of the project — it is just a
convenient, nontrivial, nonconvex function to optimize. The point is the
**optimization problem**: the decision variables, the objective, and three
specific constraints, and what happens to the solution as you change them.

### The optimization problem

- **Decision variables**: `w` = all the weights and biases of a small
  1-hidden-layer network, flattened into one vector
  (`W1, b1, W2, b2` → `w ∈ R^n`).
- **Objective**: minimize mean-squared training error
  `f(w) = (1/N) Σ (f(xᵢ; w) − yᵢ)²`.
- **Constraints** (each independently switchable):
  1. **Lipschitz bound** (bilinear, nonconvex):
     `‖W1‖_F² · ‖W2‖_F² ≤ L_max²` — caps how fast the network's output can
     change, a hard cap on model "sensitivity"/complexity.
  2. **Norm-ball** (convex quadratic): `‖w‖₂² ≤ B_max²` — the constrained
     analogue of L2 regularization.
  3. **Symmetry-breaking** (linear): `b1[0] ≤ b1[1] ≤ ... ≤ b1[H-1]` —
     removes the relabeling symmetry of hidden units (every permutation of
     hidden neurons is an equivalent solution; this kills the redundancy).
- **Solvers compared**: IPOPT (interior point), SQP (sequential quadratic
  programming, via CasADi's `sqpmethod`), and Adam (unconstrained gradient
  descent — the baseline everyone already uses).

### The four research questions (and what we found)

1. **IPOPT vs SQP vs Adam** — same network, same data: which gets a better
   fit, and at what computational cost?
2. **Lipschitz bound sweep** — as `L_max` tightens, how does the
   train/test MSE trade-off move? (This is the generalization-vs-fit
   curve — the "money plot" of the project.)
3. **NLP size scaling** — how do solve time and IPOPT iteration count grow
   as the network (and hence the NLP) gets bigger?
4. **Noise robustness** — under noisy labels, does the hard-constrained
   solve generalize better than unconstrained Adam?

The data itself is synthetic and "teacher-student": a small fixed random
network generates the ground-truth curve, we add Gaussian noise, and the
student network (the one we optimize) has to recover it. This keeps the
regression problem simple and 1-D so you can literally see the fit on a
plot, while still being a real nonconvex optimization problem.

---

## Folder structure

```
nn_constrained_nlp/
├── README.md                 ← this file
├── requirements.txt
├── configs/
│   ├── __init__.py
│   └── experiments.py        ← the ONLY file you need to edit to add/change experiments
├── src/
│   ├── data.py               ← synthetic teacher-student dataset generator
│   ├── model.py               ← network forward pass (NumPy + CasADi symbolic), flatten/unflatten
│   ├── constraints.py         ← Lipschitz / norm-ball / symmetry-breaking constraint builders
│   ├── nlp_builder.py         ← assembles the CasADi NLP (decision vars, objective, constraints)
│   ├── baseline_adam.py       ← unconstrained Adam optimizer (uses CasADi for the gradient)
│   ├── solver.py               ← solve(): dispatches to IPOPT / SQP / Adam, unified result format
│   ├── analysis.py             ← Lipschitz estimate, constraint-violation metrics
│   ├── logger.py                ← saves results/*.json + results/summary.csv
│   └── plotting.py              ← every figure
├── results/                   ← auto-generated JSON + summary.csv (git-ignored)
├── figures/                   ← auto-generated PNGs (git-ignored)
├── tests/
│   ├── test_model.py           ← flatten/unflatten roundtrip, NumPy==CasADi forward pass check
│   └── test_constraints.py     ← each constraint computes the value/bounds you'd expect by hand
└── main.py                     ← CLI entry point
```

This mirrors the structure of the double-pendulum project (`files/`) on
purpose, so both projects look like the same "house style": all tunable
numbers live in `configs/`, all logic lives in `src/`, and `results/` /
`figures/` are throwaway, regenerable outputs.

---

## How to run it (step by step)

**1. Install dependencies** (from inside `nn_constrained_nlp/`):
```bash
pip install -r requirements.txt
```
(If you already have `casadi`, `numpy`, `scipy`, `matplotlib` installed
globally — e.g. from the double-pendulum project — you don't need to do
anything else.)

**2. Run the unit tests** (sanity check that the math is wired correctly):
```bash
python -m pytest tests/ -v
```
Expect: `10 passed` in a few seconds.

**3. See what would run, without running it:**
```bash
python main.py --dry-run
```
Expect: a list of 92 experiments across 13 groups
(`method_comparison`, `lipschitz_sweep`, `size_scaling`, `noise_robustness`,
`multistart`, `kkt_analysis`, `penalty_vs_hard`, `warm_start`,
`constraint_geometry`, `convergence_rate`, `hessian_comparison`,
`complexity`, `pendulum`).

**4. Run everything:**
```bash
python main.py
```
Expect: ~1-2 minutes total wall time, a results table printed at the end,
~100 JSON files + `summary.csv` in `results/`, and ~110 PNGs in `figures/`.

**5. Run just one group** (useful while iterating):
```bash
python main.py --group lipschitz_sweep
python main.py --group method_comparison
python main.py --group size_scaling
python main.py --group noise_robustness
```

**6. Run a single experiment:**
```bash
python main.py --name exp_method_ipopt
```

**7. Skip figure generation** (faster, e.g. while debugging the solver):
```bash
python main.py --no-plots
```

**8. Build the presentation animations** (after `python main.py` has run at
least once — two of the four animations replay stored results):
```bash
python animate.py                 # all four GIFs into figures/
python animate.py --only race     # or one at a time: intro / sweep / race / complexity
```
Expect: 4 GIFs in `figures/` (`anim_problem_intro`, `anim_lipschitz_sweep`,
`anim_solver_race`, `anim_complexity`), ~1-2 minutes total. They animate when
opened in a browser or in a running PowerPoint slideshow.

---

## What results you should see

Numbers will vary very slightly by machine/IPOPT version, but the
*pattern* should always look like this:

### Group 1 — method comparison (`fig_method_comparison.png`)
IPOPT and SQP land on essentially the same solution (train MSE ≈ 0.0018,
test MSE ≈ 0.0031) because they're solving the same constrained NLP to
the same first-order optimality conditions. Adam (unconstrained) gets a
slightly different point (train MSE ≈ 0.0022, test MSE ≈ 0.0029) because
it isn't solving the same problem — there's no constraint to satisfy.
On **cost**: IPOPT converges in ~250-300 iterations and well under a
second; SQP needs ~2000+ iterations (it's using a BFGS Hessian
approximation so it has only first-order curvature information) and is
noticeably slower; Adam takes its full 3000 iterations every time
(fixed budget, no stopping criterion beyond a tiny tolerance).

### Group 2 — Lipschitz sweep (`fig_lipschitz_sweep.png`)
This is the key plot. As `L_max` grows from 0.5 to 32:
- **train MSE goes down** (~0.010 → ~0.005): more capacity, better fit.
- **test MSE goes up** (~0.011 → ~0.018): the network starts overfitting
  once it's no longer capacity-limited.
- The right-hand panel shows the *achieved* `‖W1‖_F·‖W2‖_F` tracks the
  bound almost exactly at every level — i.e. the constraint is **active**
  (binding) the whole way, confirming it's actually doing something, not
  just sitting there unused.

### Group 3 — size scaling (`fig_size_scaling.png`)
Solve time grows from ~0.15s (H=4, 13 weights) to ~7s (H=64, 193 weights)
— clearly superlinear, since IPOPT's per-iteration cost involves a
Hessian factorization that grows with problem size. Iteration count grows
much more mildly (roughly 130 → 330). This is the evidence for "how does
solver cost scale with NLP dimension."

### Group 4 — noise robustness (`fig_noise_robustness.png`)
At noise σ=0 both methods get essentially zero error (nothing to overfit
to). As σ increases to 0.1 and 0.3, both methods' test MSE grows, and the
constrained (IPOPT) and unconstrained (Adam) solutions stay close to each
other — with the defaults in `configs/experiments.py`, the constraints
are loose enough that this isn't a dramatic difference. **If you want a
more dramatic story for the report**, tighten `L_max`/`B_max` for this
group in `configs/experiments.py` (e.g. `L_max=1.5`) and re-run — a
tighter cap should widen the gap in IPOPT's favor at high noise.

### Group 10 — convergence rate (`fig_convergence_rate.png`)
KKT residual per iteration, same objective / data / initial guess for all
four runs. Adam (first-order) stalls at a residual of ~2e-4 after its full
3000 iterations. Gauss-Newton/Levenberg-Marquardt (self-implemented in
`src/gauss_newton.py`, course notes §6.6–6.7) crashes to 1e-4 within ~15
Jacobian-cheap iterations — territory Adam never reaches — then stalls at
its theory-predicted floor (nonzero residuals + singular JᵀJ ⇒ linear rate,
LM damping mandatory). IPOPT (Newton-type) reaches 1e-6 on the same
unconstrained problem, and on the Lipschitz-constrained problem shows the
textbook **superlinear tail** — the residual plunges from 1e-5 to 1e-9 in
the last ~3 iterations (iteration ~150). One subtlety worth reporting: the
*unconstrained* problem's minimizers are non-isolated (hidden-unit
permutation symmetry + overparameterization ⇒ singular Hessian), so even
IPOPT loses its superlinear rate there and needs a relaxed tolerance
(1e-6); the active constraints remove enough degeneracy to restore it.

### Group 11 — exact vs. L-BFGS Hessian (`fig_hessian_comparison.png`)
The same constrained NLP at every network size, solved with IPOPT's exact
(AD) Hessian vs. its limited-memory BFGS approximation, both at tol=1e-4
(the tolerance both can attain — with L-BFGS, IPOPT cannot reach even 1e-5
in 5000 iterations on this degenerate landscape, which is itself the
headline finding). Exact: 23–41 iterations at every size. L-BFGS: 328–4347
iterations and 10–300× more wall time. On a dense ~200-variable NLP the
per-iteration savings of skipping the Hessian never pay for the lost
curvature information.

### Group 12 — complexity race (`fig_complexity.png`, `anim_complexity.gif`)
Adam and the self-implemented Gauss-Newton/LM run at the same sizes and data
as the size-scaling group (H = 4→64), so the scaling story becomes a
comparison. Cost: Adam stays nearly flat while both exact methods grow
steeply (each factorizes an O(n³) dense system). Generalization (the
punchline): unconstrained GN/LM overfits as capacity grows — train MSE
halves while test MSE deteriorates 0.0027 → 0.0044; Adam is protected only
by its own stall; the constrained IPOPT solve stays at test 0.0031 with
violation 0 at every size — protected by design, not by accident.

### Group 13 — pendulum system ID (`fig_pendulum.png`)
The same machinery on a physically meaningful task: identify a damped
pendulum's free response theta(t) [rad] from simulated noisy angle
measurements
(omega = 2 rad/s, zeta = 0.15, t in [0, 6] s). The Lipschitz bound acquires
physical units — it certifies |d theta_hat/dt| <= L_max rad/s. The certified
fit (L_max = 4) matches/beats unconstrained Adam (test MSE 0.0029 vs 0.0032)
while carrying the guarantee; the over-tight run (L_max = 1, below the true
~2 rad/s) shows a certified underfit — the constraint dial in rad/s.

### Per-experiment fit plots
Every experiment also gets its own `figures/<name>.png`: training/test
points scattered against the learned curve. These are good "proof it
actually works" figures for slides — you can visually see the fit
tighten or loosen as you change `L_max`.

---

## Editing experiments

Everything you'd want to change lives in `configs/experiments.py`:
`H` (network size), `L_max`/`B_max` (constraint tightness), `noise_std`,
which constraints are switched on, IPOPT/SQP/Adam solver settings. Change
a value, set `enabled=False` on anything you don't want to run, and
re-run `python main.py`.
