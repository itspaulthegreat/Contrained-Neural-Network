"""
=============================================================
  EXPERIMENT CONFIGURATION FILE
  Neural Network Weight Optimization as a Constrained NLP
=============================================================

This is the ONLY file you need to touch to run experiments.

HOW TO USE:
  - Each entry in EXPERIMENTS is one experiment run.
  - Set enabled=True/False to turn experiments on/off.
  - Change any parameter and re-run main.py.
  - Results are saved automatically to results/ and figures/.

PARAMETER REFERENCE:
  method      : 'ipopt' (interior point, constrained)
                'sqp'   (sequential quadratic programming, constrained)
                'adam'  (unconstrained gradient descent baseline)
  H           : number of hidden units in the student network
  L_max       : Lipschitz bound  ||W1||_F * ||W2||_F <= L_max  (only used if use_lipschitz)
  B_max       : weight norm-ball radius  ||w||_2 <= B_max      (only used if use_norm_ball)
  s1_max/s2_max : exact spectral-norm caps ||W1||_2<=s1_max, ||W2||_2<=s2_max
                  (only used if use_spectral_norm)
  use_lipschitz / use_norm_ball / use_symmetry_break / use_spectral_norm
              : turn each constraint on/off independently
  noise_std   : std of Gaussian noise added to the synthetic labels
  seed        : random seed (data generation + initial weight guess)
"""

import numpy as np

# ── Data generation (synthetic teacher-student regression) ───────────────────
DATA = dict(
    d_in=1, d_out=1,
    H_teacher=6,            # hidden units of the fixed random "ground truth" network
    n_train=60,
    n_test=40,
    x_range=(-3.0, 3.0),
)

# ── Default network / constraint settings (overridden per-experiment) ────────
DEFAULTS = dict(
    H=8,
    L_max=4.0,
    B_max=6.0,
    s1_max=2.0,             # spectral-norm cap on W1  ||W1||_2 <= s1_max
    s2_max=2.0,             # spectral-norm cap on W2  ||W2||_2 <= s2_max
    use_lipschitz=False,
    use_norm_ball=False,
    use_symmetry_break=False,
    use_spectral_norm=False,  # exact spectral-norm bound (off by default)
    noise_std=0.05,
    seed=0,
    init_scale=0.5,
)

# ── Solver settings ────────────────────────────────────────────────────────
IPOPT_OPTS = dict(
    ipopt=dict(max_iter=2000, tol=1e-8, print_level=0),
    print_time=False,
)
SQP_OPTS = dict(
    qpsol='qrqp',
    max_iter=3000,
    hessian_approximation='limited-memory',  # BFGS -- keeps QP subproblems convex;
                                              # the exact Hessian is indefinite here
                                              # because the Lipschitz constraint is bilinear
    tol_pr=1e-7,
    tol_du=1e-5,
    print_time=False,
    print_iteration=False,
    print_header=False,
    print_status=False,
    qpsol_options=dict(print_iter=False, print_header=False, error_on_fail=False),
)

ADAM_OPTS = dict(
    lr=0.02, n_iter=3000, beta1=0.9, beta2=0.999,
)


def _make(name, label, group, **overrides):
    cfg = dict(DATA)
    cfg.update(DEFAULTS)
    cfg.update(
        name=name, label=label, group=group, enabled=True,
        ipopt_opts=IPOPT_OPTS, sqp_opts=SQP_OPTS, adam_opts=ADAM_OPTS,
    )
    cfg.update(overrides)
    return cfg


# =============================================================================
#  EXPERIMENTS — edit this list freely
# =============================================================================
EXPERIMENTS = []

# ── GROUP 1: method comparison ────────────────────────────────────────────
# Same network, same data, same constraints (where applicable) -- only the
# optimization algorithm changes. Answers: how do IPOPT / SQP / Adam compare
# in solution quality and computational cost on the identical problem?
_method_common = dict(
    H=8, use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
    L_max=4.0, B_max=6.0, noise_std=0.05,
)
EXPERIMENTS += [
    _make('exp_method_ipopt', 'Method comparison — IPOPT (constrained)',
          'method_comparison', method='ipopt', **_method_common),
    _make('exp_method_sqp', 'Method comparison — SQP (constrained)',
          'method_comparison', method='sqp', **_method_common),
    _make('exp_method_adam', 'Method comparison — Adam (unconstrained)',
          'method_comparison', method='adam', H=8, noise_std=0.05),
]

# Exact spectral-norm bound vs. the Frobenius-proxy Lipschitz bound, at the
# SAME capacity budget. The Frobenius bound above caps the *product*
# ||W1||_F * ||W2||_F <= L_max=4.0; here each layer's true spectral norm is
# capped separately (||W1||_2 <= 2, ||W2||_2 <= 2, so the gain product is also
# <= 4). Since ||.||_2 <= ||.||_F, the spectral bound is the tighter, exact
# version of the same Lipschitz-gain budget -- this experiment shows whether
# that tightening helps or hurts MSE. (norm-ball + symmetry breaking are kept
# on to match the other constrained method_comparison runs.)
EXPERIMENTS.append(_make(
    'exp_method_spectral', 'Method comparison — IPOPT (exact spectral-norm bound)',
    'method_comparison', method='ipopt', H=8,
    use_lipschitz=False, use_spectral_norm=True, use_norm_ball=True,
    use_symmetry_break=True, s1_max=2.0, s2_max=2.0, B_max=6.0, noise_std=0.05,
))

# ── GROUP 2: Lipschitz bound sweep ────────────────────────────────────────
# Only the Lipschitz constraint is active (isolated from the other two) so
# its effect on the train/test trade-off is visible on its own. Answers:
# how does tightening L_max trade off training fit against generalization?
for L in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    tag = str(L).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_lipschitz_L{tag}', f'Lipschitz sweep — L_max={L}',
        'lipschitz_sweep', method='ipopt', H=8, L_max=L,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.1,
    ))

# ── GROUP 3: NLP size scaling ─────────────────────────────────────────────
# Same constraints, growing network -> growing NLP. Answers: how does
# IPOPT's solve time and iteration count scale with problem dimension?
for H in [4, 8, 16, 32, 64]:
    EXPERIMENTS.append(_make(
        f'exp_size_H{H}', f'Size scaling — H={H} hidden units',
        'size_scaling', method='ipopt', H=H,
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
        L_max=4.0, B_max=6.0, noise_std=0.05,
    ))

# ── GROUP 4: noise robustness ─────────────────────────────────────────────
# Constrained (IPOPT, hard Lipschitz + norm-ball) vs unconstrained (Adam),
# repeated at increasing label noise. Answers: do the hard constraints
# improve generalization under noisy data compared to unconstrained descent?
for noise in [0.0, 0.1, 0.3]:
    tag = str(noise).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_noise_{tag}_ipopt', f'Noise robustness — sigma={noise} (IPOPT, constrained)',
        'noise_robustness', method='ipopt', H=8, noise_std=noise,
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
        L_max=4.0, B_max=6.0,
    ))
    EXPERIMENTS.append(_make(
        f'exp_noise_{tag}_adam', f'Noise robustness — sigma={noise} (Adam, unconstrained)',
        'noise_robustness', method='adam', H=8, noise_std=noise,
    ))

# =============================================================================
#  NEW GROUPS — deeper optimization-side studies (added on top of the
#  original four groups above; nothing above this line was changed)
# =============================================================================

# ── GROUP 5: multi-start / local minima study ─────────────────────────────
# The Lipschitz-constrained NLP is nonconvex (the constraint is bilinear,
# and the tanh network is already nonconvex on its own). IPOPT can converge
# to different local minima depending on the initial guess. All 20 runs use
# the SAME training data (fixed data_seed) and the SAME L_max -- only the
# random initialization seed changes -- so any spread in the result is
# genuinely about the shape of the optimization landscape, not the data.
# init_scale is intentionally larger than the project default (4.0 vs 0.5):
# small random jitters near the origin all fall in the same basin, so a
# wider initial spread is needed to actually land in different basins.
# Answers: does this NLP have a real nonconvex landscape with multiple,
# distinct local optima (a concern Adam users never have to think about)?
_MULTISTART_DATA_SEED = 999
for i in range(20):
    EXPERIMENTS.append(_make(
        f'exp_multistart_seed{i}', f'Multi-start — init seed {i}',
        'multistart', method='ipopt', H=8, L_max=4.0,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.05, seed=i, data_seed=_MULTISTART_DATA_SEED, init_scale=4.0,
    ))

# ── GROUP 6: KKT / dual-variable analysis ─────────────────────────────────
# Same Lipschitz sweep as 'lipschitz_sweep' (same H, same noise_std, same L
# range) but additionally records IPOPT's Lagrange multiplier (dual
# variable / "shadow price") for the Lipschitz constraint at each L_max.
# Answers: at which point does the constraint actually start binding
# (lambda > 0, costly) versus sit slack (lambda ~ 0, free)?
for L in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    tag = str(L).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_kkt_L{tag}', f'KKT analysis — L_max={L}',
        'kkt_analysis', method='ipopt', H=8, L_max=L,
        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
        noise_std=0.1,
    ))

# ── GROUP 7: penalty method vs. hard constraint ───────────────────────────
# The "everyone already does this" way to handle a constraint in ML is a
# penalty term in the loss, minimize f(w) + rho*violation(w), with plain
# Adam ('penalty_adam' method, see src/penalty_adam.py). Compared against
# IPOPT, which enforces the identical Lipschitz constraint exactly (zero
# violation by construction). Answers: how much does an unconstrained
# penalty method actually violate the constraint, and how does that trade
# off against fit, as rho varies from very loose to very strict?
#
# L_max=1.0 here (NOT the project default 4.0): the network's natural,
# unpenalized optimum for this data sits at Lipschitz~1.6, i.e. *below*
# L_max=4.0 -- so at the default bound the constraint never binds and the
# violation is 0 for every rho, which would make for a flat, uninteresting
# plot. L_max=1.0 sits below that natural optimum, so the constraint
# actually has something to fight against.
#
# rho range is empirically chosen rather than a naive log-decade sweep
# (0.1 .. 1000): Adam's per-parameter adaptive step normalization (it
# normalizes by the RMS of past gradients) means the penalty's *direction*
# matters far more than its magnitude, so enforcement saturates extremely
# fast -- the transition from "ignores the constraint" to "fully enforces
# it" happens between rho=1e-5 and rho=1e-4, not between 0.1 and 1000.
# rho=0.1..1000 all land in the already-saturated regime and look
# identical to each other; this range is the one that actually shows the
# trade-off the experiment is meant to demonstrate.
_PENALTY_COMMON = dict(H=8, L_max=1.0, noise_std=0.05,
                        use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False)
EXPERIMENTS.append(_make(
    'exp_penalty_ipopt_ref', 'Penalty comparison — IPOPT (hard constraint, reference)',
    'penalty_vs_hard', method='ipopt', **_PENALTY_COMMON,
))
for rho in [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1.0]:
    tag = str(rho).replace('.', 'p').replace('-', 'm')
    EXPERIMENTS.append(_make(
        f'exp_penalty_rho{tag}', f'Penalty comparison — Adam, rho={rho:g}',
        'penalty_vs_hard', method='penalty_adam', rho=rho, **_PENALTY_COMMON,
    ))

# ── GROUP 8: warm-start study ─────────────────────────────────────────────
# A single driver entry (handled specially in main.py, see src/warm_start.py):
# it runs the SAME Lipschitz sweep twice over a tightening sequence of L_max
# (32 -> 0.5), once cold-starting every solve from the same random w0 and once
# warm-starting each solve from the previous (looser) solution. Answers: how
# much does warm starting cut IPOPT's iteration count when a constraint is
# tightened incrementally? (The classic continuation / homotopy trick.)
EXPERIMENTS.append(_make(
    'exp_warm_start', 'Warm-start study — incremental Lipschitz tightening',
    'warm_start', method='ipopt', H=8,
    use_lipschitz=True, use_norm_ball=False, use_symmetry_break=False,
    noise_std=0.1,
    warm_start_L_values=[32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5],  # tightening order
))

# ── GROUP 9: constraint geometry / interaction ────────────────────────────
# Lipschitz bound fixed (L_max=4.0), norm-ball radius B_max swept from tight
# (0.5) to loose (20.0), BOTH constraints active. Each solve records both
# dual variables (which constraint binds) plus train/test MSE and the worst
# violation. Answers: where does control pass from the norm-ball (tight B_max)
# to the Lipschitz bound (loose B_max)? Dispatched to constraint_geometry.
for B in [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]:
    tag = str(B).replace('.', 'p')
    EXPERIMENTS.append(_make(
        f'exp_geom_B{tag}', f'Constraint geometry — B_max={B}',
        'constraint_geometry', method='ipopt', H=8, L_max=4.0, B_max=B,
        use_lipschitz=True, use_norm_ball=True, use_symmetry_break=False,
        noise_std=0.1,
    ))

# ── GROUP 10: convergence rate (KKT residual vs. iteration) ───────────────
# IPOPT is a Newton-type method: near the solution it uses exact second-order
# information, so its KKT residual drops superlinearly in the final
# iterations. Adam is first-order: its gradient norm decays slowly and
# stalls. The IPOPT-unconstrained and Adam runs minimize the IDENTICAL
# unconstrained MSE from the identical w0, so for both the plotted residual
# is the same quantity (the stationarity residual ||grad f||) -- a fair,
# same-problem comparison of convergence *rate*, not just final MSE.
# The Lipschitz-constrained IPOPT run (L_max=1.0, chosen so the constraint
# binds -- the unconstrained optimum sits at Lipschitz~1.6) is added to show
# the same superlinear tail survives an active nonconvex constraint.
#
# The unconstrained run gets tol=1e-6 instead of the project-default 1e-8:
# the unconstrained problem's minimizers are non-isolated (hidden units can
# be permuted, and the H=8 student is overparameterized for the H=6 teacher),
# so the Hessian is singular at every minimizer and Newton-type superlinear
# convergence is lost -- the residual dips to ~1e-8 but oscillates and never
# stays there, exhausting max_iter. 1e-6 is attainable (reached at iteration
# ~780) and still ~100x below where Adam stalls after 3000 iterations
# (~7e-5). The constrained run keeps 1e-8: the active constraints remove
# enough of the degeneracy that IPOPT reaches it in ~150 iterations.
_CONV_COMMON = dict(H=8, noise_std=0.05, seed=0)
EXPERIMENTS += [
    _make('exp_conv_ipopt_unc', 'Convergence rate — IPOPT (unconstrained)',
          'convergence_rate', method='ipopt',
          ipopt_opts=dict(ipopt=dict(max_iter=2000, tol=1e-6, print_level=0),
                          print_time=False),
          **_CONV_COMMON),
    _make('exp_conv_ipopt_con', 'Convergence rate — IPOPT (Lipschitz-constrained)',
          'convergence_rate', method='ipopt', use_lipschitz=True, L_max=1.0,
          **_CONV_COMMON),
    _make('exp_conv_adam', 'Convergence rate — Adam (unconstrained)',
          'convergence_rate', method='adam', **_CONV_COMMON),
    # Gauss-Newton / Levenberg-Marquardt (course notes §6.6-6.7), implemented
    # from scratch in src/gauss_newton.py. The objective IS a nonlinear
    # least-squares problem, so the course's dedicated fitting method belongs
    # in this comparison: it uses the exact residual Jacobian (AD) but drops
    # the second-order residual term -- structurally between Adam
    # (first-order) and IPOPT's exact Newton. LM damping is required because
    # JtJ is singular here (overparameterization + permutation symmetry).
    _make('exp_conv_gn', 'Convergence rate — Gauss-Newton/LM (unconstrained)',
          'convergence_rate', method='gauss_newton',
          gn_opts=dict(max_iter=200, lam0=1e-3, tol=1e-8),
          **_CONV_COMMON),
]

# ── GROUP 11: exact vs. limited-memory (L-BFGS) Hessian in IPOPT ──────────
# Same constrained problem as 'size_scaling' at every H, solved twice: once
# with the exact Hessian of the Lagrangian (CasADi automatic differentiation,
# IPOPT's default here) and once with IPOPT's L-BFGS approximation. Exact
# Hessian: expensive per iteration (build + factorize a dense n x n matrix)
# but locally superlinear -> few iterations. L-BFGS: cheap per iteration but
# only first-order curvature -> more iterations. Answers: at which problem
# size (if any) does the cheaper-per-iteration approximation win on total
# solve time, and what does it cost in iteration count?
#
# Both variants use tol=1e-4 -- NOT the project default 1e-8 -- because the
# comparison is only fair at a tolerance both can attain. With the exact
# Hessian IPOPT reaches 1e-8 in ~300 iterations, but with L-BFGS it cannot
# reach even 1e-5 within 5000 iterations on this problem (the NN training
# landscape is too degenerate for first-order curvature estimates). That
# gap is itself the headline result of this group; the tol=1e-4 runs then
# quantify the iteration/time cost at the accuracy L-BFGS can deliver.
_HESS_IPOPT_EXACT = dict(ipopt=dict(max_iter=5000, tol=1e-4, print_level=0),
                          print_time=False)
_HESS_IPOPT_LBFGS = dict(ipopt=dict(max_iter=5000, tol=1e-4, print_level=0,
                                     hessian_approximation='limited-memory'),
                          print_time=False)
for H in [4, 8, 16, 32, 64]:
    for mode, tag, opts in [('exact', 'exact', _HESS_IPOPT_EXACT),
                            ('limited-memory', 'lbfgs', _HESS_IPOPT_LBFGS)]:
        EXPERIMENTS.append(_make(
            f'exp_hess_{tag}_H{H}', f'Hessian comparison — {mode}, H={H}',
            'hessian_comparison', method='ipopt', H=H,
            use_lipschitz=True, use_norm_ball=True, use_symmetry_break=True,
            L_max=4.0, B_max=6.0, noise_std=0.05,
            ipopt_opts=opts, hessian_mode=mode,
        ))

# ── GROUP 12: complexity race — solvers vs baselines as the network grows ──
# Group 3 measures how IPOPT alone scales; this group adds the baselines at
# the identical sizes and data (noise 0.05, seed 0, H = 4..64) so the scaling
# claim becomes a COMPARISON, not a solo measurement: exact constrained
# Newton (from Group 3) vs unconstrained Adam vs self-implemented
# Gauss-Newton/LM. Answers: as the NLP grows, who pays what — and what do
# you get for it? (IPOPT: O(n^3) factorizations but certified constraints
# and 1e-8 optimality; Adam: flat cheap iterations but stalls and knows no
# constraints; GN/LM: Jacobian-cheap and fast to medium accuracy, then the
# residual-limited floor.)
for H in [4, 8, 16, 32, 64]:
    EXPERIMENTS.append(_make(
        f'exp_complexity_adam_H{H}', f'Complexity race — Adam, H={H}',
        'complexity', method='adam', H=H, noise_std=0.05,
    ))
    EXPERIMENTS.append(_make(
        f'exp_complexity_gn_H{H}', f'Complexity race — Gauss-Newton/LM, H={H}',
        'complexity', method='gauss_newton', H=H, noise_std=0.05,
        gn_opts=dict(max_iter=200, lam0=1e-3, tol=1e-8),
    ))

# ── GROUP 13: physical task — pendulum system identification ───────────────
# Answers the "what is it even learning?" question with a physically
# meaningful task: fit the free response theta(t) [rad] of a damped pendulum
# (omega = 2 rad/s, zeta = 0.15) from noisy angle measurements over t in
# [0, 6] s -- classical 1-D system identification. Here the Lipschitz bound
# has a physical reading: it caps |d theta_hat / d t|, i.e. the maximum
# angular velocity the fitted model can represent. The true response has
# |theta'| <= ~2 rad/s; since the Frobenius product over-estimates the true
# sensitivity (||.||_2 <= ||.||_F), L_max = 4 leaves the physics enough room
# to fit accurately while still certifying |d theta_hat/dt| <= 4 rad/s.
# L_max = 1 is deliberately BELOW the physical rate to show what an
# over-tight certificate does (visible, certified underfit). Adam is the
# unconstrained baseline.
_PEND_COMMON = dict(task='pendulum', H=8, noise_std=0.05, seed=0,
                    x_range=(0.0, 6.0))
EXPERIMENTS += [
    _make('exp_pendulum_ipopt', 'Pendulum system ID — IPOPT, L_max=4 (certified)',
          'pendulum', method='ipopt', use_lipschitz=True, L_max=4.0,
          use_symmetry_break=True, **_PEND_COMMON),
    _make('exp_pendulum_tight', 'Pendulum system ID — IPOPT, L_max=1 (over-tight)',
          'pendulum', method='ipopt', use_lipschitz=True, L_max=1.0,
          use_symmetry_break=True, **_PEND_COMMON),
    _make('exp_pendulum_adam', 'Pendulum system ID — Adam (unconstrained)',
          'pendulum', method='adam', **_PEND_COMMON),
]
