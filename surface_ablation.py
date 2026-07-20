"""
surface_ablation.py
─────────────────────
The objective landscape and the constraints, watched live: a 3-D surface of
the training objective f(w) on a 2-D slice through the 25-dimensional weight
space, with the constraints switched on one at a time (an ablation) and the
optimum moving in response.

Honesty notes (baked into the figure):
  * f lives in R^25; what is drawn is f restricted to the 2-D plane through
    the CONSTRAINED stage optima -- an illustrative slice, not the whole
    landscape.
  * The unconstrained optimum sits at ||w|| ~ 8000, far outside any readable
    frame around the constrained optima (||w|| ~ 12-15). It is indicated with
    an arrow + its numbers instead of being plotted -- squeezing it into the
    frame would flatten the entire surface (z would span ~1e7).

Stages (cumulative, each solved by IPOPT on the same data; sigma = 0.2, seed 2):
    1  unconstrained                                  (off-frame, annotated)
    2  + Lipschitz      ||W1||_F^2 ||W2||_F^2 <= L^2  (L = 1.5)
    3  + norm ball      ||w||_2^2 <= B^2              (B chosen ACTIVE: 0.8*||w2||)
    4  + symmetry       b1[j] <= b1[j+1]

Outputs:
    figures/anim_surface_ablation.gif   -- the live ablation (for the talk)
    figures/fig_surface_ablation.png    -- static composite (for the report)

    python surface_ablation.py          (~1 min)
"""

import os
import numpy as np
import casadi as ca

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim

from src.data import generate_dataset
from src.model import (param_shapes, n_params, forward_symbolic,
                       mse_numpy, random_init, unflatten_symbolic, unflatten_numpy)
from src.constraints import lipschitz_constraint

FIG = os.path.join(os.path.dirname(__file__), 'figures')
SEED, SIGMA, H = 2, 0.2, 8
L_MAX = 1.5

plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 130})


def solve(Xtr, ytr, use_lip=False, use_ball=False, B=None, use_sym=False):
    shapes = param_shapes(1, H, 1)
    n = n_params(shapes)
    w = ca.MX.sym('w', n)
    f = ca.sumsqr(forward_symbolic(w, Xtr, shapes) - ytr) / Xtr.shape[1]
    gs, lbs, ubs = [], [], []
    W1, b1, W2, b2 = unflatten_symbolic(w, shapes)
    if use_lip:
        g, lb, ub = lipschitz_constraint(W1, W2, L_MAX)
        gs.append(g); lbs.append(lb); ubs.append(ub)
    if use_ball:
        gs.append(ca.sumsqr(w)); lbs.append(-np.inf); ubs.append(float(B ** 2))
    if use_sym:
        for j in range(H - 1):
            gs.append(b1[j] - b1[j + 1]); lbs.append(-np.inf); ubs.append(0.0)
    if gs:
        nlp = {'x': w, 'f': f, 'g': ca.vertcat(*gs)}
        args = dict(lbg=np.array(lbs, float), ubg=np.array(ubs, float))
    else:
        nlp, args = {'x': w, 'f': f}, {}
    s = ca.nlpsol('s', 'ipopt', nlp,
                  dict(ipopt=dict(max_iter=3000, tol=1e-8, print_level=0),
                       print_time=False))
    sol = s(x0=random_init(shapes, scale=0.5, seed=SEED), **args)
    return np.asarray(sol['x']).flatten()


def main():
    shapes = param_shapes(1, H, 1)
    Xtr, ytr, _, _ = generate_dataset(n_train=40, n_test=40, noise_std=SIGMA, seed=SEED)
    mse = lambda w: mse_numpy(w, Xtr, ytr, shapes)

    # ---- the four cumulative stages ----
    w1 = solve(Xtr, ytr)
    w2 = solve(Xtr, ytr, use_lip=True)
    B_demo = round(float(np.linalg.norm(w2)) * 0.8, 2)       # ball chosen ACTIVE
    w3 = solve(Xtr, ytr, use_lip=True, use_ball=True, B=B_demo)
    w4 = solve(Xtr, ytr, use_lip=True, use_ball=True, B=B_demo, use_sym=True)
    stages = [
        ('1 · unconstrained', w1, 'no constraints'),
        (f'2 · + Lipschitz  (L = {L_MAX})', w2, f'Lipschitz ON (L={L_MAX})'),
        (f'3 · + norm ball  (B = {B_demo})', w3, f'Lipschitz + ball (B={B_demo}, active)'),
        ('4 · + symmetry', w4, 'Lipschitz + ball + symmetry'),
    ]
    for name, w, _ in stages:
        W1m, _, W2m, _ = unflatten_numpy(w, shapes)
        print(f'{name:28s} f(w*) = {mse(w):.4f}   ||W1||*||W2|| = '
              f'{np.linalg.norm(W1m)*np.linalg.norm(W2m):.3f}   ||w|| = {np.linalg.norm(w):.2f}')

    # ---- 2-D plane through the CONSTRAINED optima (stages 2-4 only) ----
    # Stage 1 sits at ||w|| ~ 8000: including it would make the frame span
    # thousands of units and flatten the surface into unreadability.
    Wm = np.stack([w2, w3, w4])
    center = Wm.mean(axis=0)
    U, S, Vt = np.linalg.svd(Wm - center, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    def proj(w):
        return ((w - center) @ e1, (w - center) @ e2)
    coords = {1: proj(w1), 2: proj(w2), 3: proj(w3), 4: proj(w4)}

    span = 1.7 * max(max(abs(a), abs(b)) for k, (a, b) in coords.items() if k != 1)
    span = max(span, 2.0)
    g = np.linspace(-span, span, 61)
    A, Bg = np.meshgrid(g, g)
    Z = np.empty_like(A)
    lip_ok = np.empty_like(A, dtype=bool)   # Lipschitz feasible at this grid point
    ball_ok = np.empty_like(A, dtype=bool)  # norm-ball feasible
    sym_ok = np.empty_like(A, dtype=bool)   # bias ordering feasible
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            wp = center + A[i, j] * e1 + Bg[i, j] * e2
            Z[i, j] = mse(wp)
            W1p, b1p, W2p, _ = unflatten_numpy(wp, shapes)
            lip_ok[i, j] = np.sum(W1p ** 2) * np.sum(W2p ** 2) <= L_MAX ** 2 + 1e-9
            ball_ok[i, j] = np.sum(wp ** 2) <= B_demo ** 2 + 1e-9
            sym_ok[i, j] = np.all(np.diff(b1p.flatten()) >= -1e-9)
    Zc = np.clip(Z, None, np.percentile(Z, 96))

    # cumulative feasible region per stage (what the constraints allow so far).
    # Stage 4 (symmetry) keeps the stage-3 shading: the ordered-bias wedge has
    # measure ~zero on a generic 2-D plane, so shading it would (wrongly) paint
    # the feasible optimum gray — it is annotated in words instead.
    feas = [np.ones_like(lip_ok), lip_ok, lip_ok & ball_ok, lip_ok & ball_ok]
    pct = [100.0 * m.mean() for m in feas]
    sym_pct = 100.0 * (lip_ok & ball_ok & sym_ok).mean()
    print('feasible share of the plotted view per stage: '
          + '  '.join(f'{p:.0f}%' for p in pct)
          + f'   (symmetry wedge on this plane: {sym_pct:.1f}%)')
    captions = [
        'no constraints: 100% of this view is feasible',
        f'the Lipschitz bound keeps {pct[1]:.0f}% of this view feasible',
        f'+ the norm ball: {pct[2]:.0f}% stays feasible',
        'symmetry admits ONE bias ordering — a wedge too thin to shade in 2-D;\n'
        'the optimum jumps to an ordered equivalent (same f to 4 decimals)',
    ]
    small_caps = [f'feasible: {pct[0]:.0f}%', f'feasible: {pct[1]:.0f}%',
                  f'feasible: {pct[2]:.0f}%', 'ordering wedge (thin in 2-D)']

    zmin, zmax = Zc.min(), Zc.max()
    f1 = mse(w1)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    base_colors = cm.viridis(Normalize(zmin, zmax)(Zc))
    base_colors[..., 3] = 0.95

    def draw_stage(ax, si, azim, small=False):
        ax.clear()
        name, w, cons = stages[si]
        colors = base_colors.copy()
        colors[~feas[si]] = (0.72, 0.72, 0.72, 0.22)   # forbidden region fades out
        ax.plot_surface(A, Bg, Zc, facecolors=colors, linewidth=0,
                        antialiased=True, shade=False)
        if small:
            ax.text2D(0.5, 0.0, small_caps[si], transform=ax.transAxes,
                      fontsize=7.5, ha='center', color='dimgray')
        else:
            ax.text2D(0.5, -0.04, captions[si], transform=ax.transAxes,
                      fontsize=8.5, ha='center', color='dimgray')
        if si == 0:
            # the unconstrained optimum is off-frame: state it, no fake marker
            ax.text2D(0.5, 0.86 if small else 0.9,
                      'optimum lies FAR outside this slice\n'
                      f'(||w|| = {np.linalg.norm(w1):.0f},  f = {f1:.4f})',
                      transform=ax.transAxes, color='red',
                      fontsize=7.5 if small else 9, ha='center')
        else:
            ks = list(range(2, si + 2))          # stages 2..current
            cs = [coords[k] for k in ks]
            zs = [mse(stages[k - 1][1]) for k in ks]
            ax.plot([c[0] for c in cs], [c[1] for c in cs], zs,
                    'o-', color='red', ms=5, lw=1.5)
            a, b = coords[si + 1]
            ax.scatter([a], [b], [mse(w)], color='red', s=150, marker='*',
                       edgecolor='white', zorder=10)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=30, azim=azim)
        ax.set_zlabel('objective f(w)', fontsize=8 if small else 10)
        if small:
            ax.set_title(f'{name}\nf(w*) = {mse(w):.4f}', fontsize=9)
        else:
            # axes are explained in the suptitle; short labels avoid colliding
            # with the bottom caption
            # u1,u2 are COORDINATES IN A PLANE, not individual weights
            ax.set_xlabel('u₁  [weight units]', fontsize=9)
            ax.set_ylabel('u₂  [weight units]', fontsize=9)
            ax.set_title(f'{name}    f(w*) = {mse(w):.4f}\nactive: {cons}',
                         fontsize=10)

    # ---- animation ----
    HOLD = 22
    frames = HOLD * len(stages) + 12
    fig = plt.figure(figsize=(9.2, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top=0.76, bottom=0.09)

    def update(k):
        si = min(k // HOLD, len(stages) - 1)
        draw_stage(ax, si, azim=-62 + 20 * (k / frames))
        fig.suptitle('The objective does NOT change — the constraints shrink the FEASIBLE region\n'
                     'u₁, u₂ are NOT single weights — they are coordinates in the plane through the '
                     'three constrained optima\n(three points fix a plane in the 25-D weight space; '
                     'distances run from their centroid)',
                     fontsize=9.5, fontweight='bold', y=0.985)
        return []

    anim = manim.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(os.path.join(FIG, 'anim_surface_ablation.gif'),
              writer=manim.PillowWriter(fps=11))
    plt.close(fig)
    print('wrote figures/anim_surface_ablation.gif')

    # ---- static composite ----
    fig = plt.figure(figsize=(12.8, 4.5))
    for si in range(len(stages)):
        ax = fig.add_subplot(1, 4, si + 1, projection='3d')
        draw_stage(ax, si, azim=-60, small=True)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.set_xlabel(''); ax.set_ylabel('')
    fig.suptitle('Constraint ablation — the objective surface stays the same; each constraint '
                 'GRAYS OUT the region it forbids\nthe view is the plane through the three constrained '
                 'optima — u₁, u₂ are coordinates in it, not single weights; the unconstrained optimum '
                 'lies far outside the frame',
                 fontsize=10.5, fontweight='bold')
    fig.text(0.5, 0.015,
             'coloured = FEASIBLE under the constraints switched on so far · gray = forbidden · '
             'stage 4: the symmetry wedge is too thin to shade in 2-D — the optimum jumps to an '
             'ordered equivalent (same f to 4 decimals; see check_symmetry_stage.py)',
             fontsize=8.5, ha='center', color='dimgray')
    fig.subplots_adjust(left=0.02, right=0.97, top=0.74, bottom=0.10, wspace=0.16)
    fig.savefig(os.path.join(FIG, 'fig_surface_ablation.png'))
    plt.close(fig)
    print('wrote figures/fig_surface_ablation.png')


if __name__ == '__main__':
    main()
