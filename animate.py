"""
animate.py
────────────
Presentation animations. Each function writes a GIF (always) and an MP4
(if ffmpeg is available) into figures/. GIFs autoplay when a PowerPoint
slideshow is running; MP4s are inserted via Insert > Video.

    python animate.py                 # build all animations
    python animate.py --only intro    # just the problem-intro animation
    python animate.py --only sweep    # just the Lipschitz-sweep animation

Anim 1  (anim_problem_intro)  — explains the task to a general audience:
        the hidden teacher curve appears, noisy samples pop in, then the
        student network visibly learns the curve (real Adam iterates,
        recorded live — not an artistic interpolation).

Anim 2  (anim_lipschitz_sweep) — the key result as a movie: the fitted
        curve stiffens as the Lipschitz bound tightens 32 → 0.5, while
        the train/test MSE trade-off builds up point by point. Uses the
        solved weight vectors already stored in results/exp_lipschitz_*.json.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import numpy as np
import casadi as ca

from src.data import generate_dataset, make_teacher, _teacher_forward
from src.model import param_shapes, forward_numpy, random_init
from src.baseline_adam import _build_grad_fn

FIGDIR = 'figures'
X_RANGE = (-3.0, 3.0)
FPS = 15

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def _save(anim, fig, name):
    """Write GIF always; MP4 too if ffmpeg is on PATH."""
    gif_path = os.path.join(FIGDIR, f'{name}.gif')
    anim.save(gif_path, writer=manim.PillowWriter(fps=FPS))
    print(f'wrote {gif_path}')
    if manim.FFMpegWriter.isAvailable():
        mp4_path = os.path.join(FIGDIR, f'{name}.mp4')
        anim.save(mp4_path, writer=manim.FFMpegWriter(fps=FPS, bitrate=1800))
        print(f'wrote {mp4_path}')
    else:
        print('(ffmpeg not found — skipped MP4, GIF is enough for PowerPoint)')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Anim 1 — the problem, explained visually (teacher → noisy data → learning)
# ─────────────────────────────────────────────────────────────────────────

def anim_problem_intro(noise_std=0.1, seed=0, H=8):
    shapes = param_shapes(1, H, 1)
    X_train, y_train, _, _ = generate_dataset(noise_std=noise_std, seed=seed)
    xs = np.linspace(X_RANGE[0], X_RANGE[1], 300).reshape(1, -1)

    teacher = make_teacher(seed=seed)
    y_teacher = _teacher_forward(xs, *teacher).flatten()

    # Real Adam run with weight snapshots (log-spaced: dense early, sparse late)
    f_fn, g_fn = _build_grad_fn(shapes, X_train, y_train)
    w = random_init(shapes, scale=0.5, seed=seed)
    n_iter = 1500
    snap_at = np.unique(np.geomspace(1, n_iter, 80).astype(int))
    snaps, m, v = [(0, w.copy())], np.zeros_like(w), np.zeros_like(w)
    for t in range(1, n_iter + 1):
        grad = np.asarray(g_fn(w)).flatten()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        w = w - 0.02 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t in snap_at:
            snaps.append((t, w.copy()))

    n_draw, n_pts = 30, 30                      # phase lengths in frames
    n_frames = n_draw + n_pts + len(snaps) + 15  # +15 = hold on final fit

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax.set_xlim(*X_RANGE)
    pad = 0.25 * (y_train.max() - y_train.min())
    ax.set_ylim(y_train.min() - pad, y_train.max() + pad)
    ax.set_xlabel('input $x$'); ax.set_ylabel('output $y$')

    teach_ln, = ax.plot([], [], color='tab:gray', lw=2.5, label='hidden ground truth (teacher)')
    scat = ax.scatter([], [], s=25, c='tab:blue', alpha=0.8, label='noisy samples (all we see)')
    fit_ln, = ax.plot([], [], color='crimson', lw=2.5, label='network fit  $f(x; w)$')
    ax.legend(loc='upper left', fontsize=9)
    title = ax.set_title('')

    order = np.argsort(X_train.flatten())  # points pop in left-to-right

    def update(k):
        if k < n_draw:                                   # phase 1: teacher draws in
            j = int(len(xs.flatten()) * (k + 1) / n_draw)
            teach_ln.set_data(xs.flatten()[:j], y_teacher[:j])
            title.set_text('An unknown function generates the data …')
        elif k < n_draw + n_pts:                         # phase 2: samples pop in
            j = int(len(order) * (k - n_draw + 1) / n_pts)
            idx = order[:j]
            scat.set_offsets(np.c_[X_train.flatten()[idx], y_train.flatten()[idx]])
            title.set_text('… we only observe noisy samples (60 points, σ = 0.1)')
        else:                                            # phase 3: the network learns
            i = min(k - n_draw - n_pts, len(snaps) - 1)
            it, wi = snaps[i]
            fit_ln.set_data(xs.flatten(), forward_numpy(wi, xs, shapes).flatten())
            title.set_text(f'training = optimizing 25 weights $w$   ·   '
                           f'iteration {it}   ·   MSE {float(f_fn(wi)):.4f}')
        return teach_ln, scat, fit_ln, title

    anim = manim.FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.tight_layout()
    _save(anim, fig, 'anim_problem_intro')


# ─────────────────────────────────────────────────────────────────────────
#  Anim 2 — Lipschitz sweep: the fit stiffens as the bound tightens
# ─────────────────────────────────────────────────────────────────────────

def anim_lipschitz_sweep():
    files = sorted(f for f in os.listdir('results')
                   if f.startswith('exp_lipschitz_') and f.endswith('.json'))
    runs = [json.load(open(os.path.join('results', f))) for f in files]
    runs.sort(key=lambda r: -r['L_max'])     # loose → tight: "turning the dial"
    if not runs:
        raise SystemExit('no results/exp_lipschitz_*.json — run: python main.py --group lipschitz_sweep')

    shapes = param_shapes(1, runs[0]['H'], 1)
    noise, seed = runs[0]['noise_std'], 0
    X_train, y_train, X_test, y_test = generate_dataset(noise_std=noise, seed=seed)
    xs = np.linspace(X_RANGE[0], X_RANGE[1], 300).reshape(1, -1)

    curves = [forward_numpy(np.array(r['w']), xs, shapes).flatten() for r in runs]
    L = [r['L_max'] for r in runs]
    tr = [r['train_mse'] for r in runs]
    te = [r['test_mse'] for r in runs]

    hold, morph = 18, 8                       # frames: pause per L, then crossfade
    per = hold + morph
    n_frames = per * (len(runs) - 1) + hold + 15

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100,
                              gridspec_kw={'width_ratios': [1.2, 1]})
    axL, axR = axes

    axL.set_xlim(*X_RANGE)
    pad = 0.25 * (y_train.max() - y_train.min())
    axL.set_ylim(y_train.min() - pad, y_train.max() + pad)
    axL.scatter(X_train.flatten(), y_train.flatten(), s=18, c='tab:blue', alpha=0.6, label='train')
    axL.scatter(X_test.flatten(), y_test.flatten(), s=18, c='tab:orange', alpha=0.6, marker='^', label='test')
    fit_ln, = axL.plot([], [], color='crimson', lw=2.5, label='constrained fit')
    axL.legend(loc='upper left', fontsize=9)
    axL.set_xlabel('input $x$'); axL.set_ylabel('output $y$')
    titleL = axL.set_title('')

    axR.set_xscale('log')
    axR.set_xlim(min(L) * 0.8, max(L) * 1.25)
    axR.set_ylim(0, max(max(tr), max(te)) * 1.15)
    axR.set_xlabel('Lipschitz bound $L_{max}$  (tightening →)')
    axR.set_ylabel('MSE')
    axR.invert_xaxis()                        # dial tightens to the right
    tr_ln, = axR.plot([], [], 'o-', color='tab:blue', label='train MSE')
    te_ln, = axR.plot([], [], 's-', color='tab:orange', label='test MSE')
    cursor = axR.axvline(L[0], color='crimson', lw=1.5, ls='--')
    axR.legend(loc='upper left', fontsize=9)
    axR.set_title('fit vs. generalization')

    def update(k):
        i, r = divmod(k, per)
        i = min(i, len(runs) - 1)
        if r < hold or i >= len(runs) - 1:    # hold on keyframe i
            y, Lc = curves[i], L[i]
            j = i
        else:                                 # crossfade to keyframe i+1 (visual only)
            a = (r - hold + 1) / morph
            y = (1 - a) * curves[i] + a * curves[i + 1]
            Lc = L[i]
            j = i
        fit_ln.set_data(xs.flatten(), y)
        act = runs[j]['lipschitz_estimate']
        titleL.set_text(f'$L_{{max}}$ = {L[j]:g}   ·   achieved '
                        f'$\\|W_1\\|_F\\|W_2\\|_F$ = {act:.4f}  (ACTIVE)')
        tr_ln.set_data(L[:j + 1], tr[:j + 1])
        te_ln.set_data(L[:j + 1], te[:j + 1])
        cursor.set_xdata([L[j], L[j]])
        return fit_ln, tr_ln, te_ln, cursor, titleL

    anim = manim.FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.suptitle('Tightening the hard Lipschitz constraint — solved exactly by IPOPT at every level')
    fig.tight_layout()
    _save(anim, fig, 'anim_lipschitz_sweep')


# ─────────────────────────────────────────────────────────────────────────
#  Anim 3 — the race: Adam (first-order) vs IPOPT (Newton-type), live
# ─────────────────────────────────────────────────────────────────────────

class _IterRecorder(ca.Callback):
    """CasADi iteration callback: stores IPOPT's primal iterate x at every
    iteration, so the evolving fit can be replayed frame by frame."""

    def __init__(self, name, nx, ng):
        ca.Callback.__init__(self)
        self.nx, self.ng = nx, ng
        self.iterates = []
        self.construct(name, {})

    def get_n_in(self):  return ca.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i):  return ca.nlpsol_out(i)
    def get_name_out(self, i): return 'ret'

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.dense(1)
        if n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        if n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        self.iterates.append(np.asarray(arg[0]).flatten().copy())
        return [0]


def anim_solver_race():
    """Adam vs Lipschitz-constrained IPOPT on the convergence-study problem
    (H=8, sigma=0.05, seed 0, L_max=1.0 binding) -- both from the SAME w0.
    Left/middle: the two fits evolving. Right: KKT residual per iteration."""
    from configs.experiments import EXPERIMENTS
    from src.nlp_builder import build_nlp

    exp_con = next(e for e in EXPERIMENTS if e['name'] == 'exp_conv_ipopt_con')
    shapes = param_shapes(1, exp_con['H'], 1)
    X_train, y_train, _, _ = generate_dataset(noise_std=exp_con['noise_std'],
                                              seed=exp_con['seed'])
    xs = np.linspace(X_RANGE[0], X_RANGE[1], 300).reshape(1, -1)

    # -- IPOPT run with iterate recording (identical NLP to the study) ------
    nlp_data = build_nlp(exp_con, X_train, y_train)
    rec = _IterRecorder('rec', nlp_data['n_vars'], nlp_data['n_constraints'])
    opts = dict(exp_con['ipopt_opts'])
    opts['iteration_callback'] = rec
    solver = ca.nlpsol('solver', 'ipopt',
                       {'x': nlp_data['w'], 'f': nlp_data['f'], 'g': nlp_data['g']}, opts)
    solver(x0=nlp_data['w0'], lbg=nlp_data['lbg'], ubg=nlp_data['ubg'])
    stats = solver.stats()
    it = stats['iterations']
    ipopt_kkt = np.maximum(np.asarray(it['inf_pr'], float),
                           np.asarray(it['inf_du'], float))
    ipopt_ws = rec.iterates
    n_ipopt = len(ipopt_ws)

    # -- Adam run from the SAME w0, recording weights + ||grad||_inf --------
    f_fn, g_fn = _build_grad_fn(shapes, X_train, y_train)
    w = np.asarray(nlp_data['w0'], float).flatten().copy()
    n_adam = 3000
    adam_kkt, adam_ws = [], {0: w.copy()}
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, n_adam + 1):
        grad = np.asarray(g_fn(w)).flatten()
        adam_kkt.append(float(np.max(np.abs(grad))))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        w = w - 0.02 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        adam_ws[t] = w.copy()
    adam_kkt = np.minimum.accumulate(adam_kkt)          # monotone envelope
    ipopt_env = np.minimum.accumulate(np.maximum(ipopt_kkt, 1e-16))

    # -- frame schedule: log-spaced over Adam's 3000 iterations -------------
    frame_its = np.unique(np.geomspace(1, n_adam, 90).astype(int))
    frame_its = np.concatenate([[0], frame_its])
    n_frames = len(frame_its) + 12                       # +12 hold at end

    C_ADAM, C_IPOPT = 'tab:green', 'tab:purple'
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=100,
                              gridspec_kw={'width_ratios': [1, 1, 1.15]})
    axA, axI, axR = axes
    pad = 0.25 * (y_train.max() - y_train.min())
    for ax, name, c in ((axA, 'Adam — first order, unconstrained', C_ADAM),
                        (axI, 'IPOPT — Newton, Lipschitz-constrained', C_IPOPT)):
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(y_train.min() - pad, y_train.max() + pad)
        ax.scatter(X_train.flatten(), y_train.flatten(), s=12, c='tab:blue', alpha=0.5)
        ax.set_xlabel('input $x$')
    axA.set_ylabel('output $y$')
    lnA, = axA.plot([], [], color=C_ADAM, lw=2.5)
    lnI, = axI.plot([], [], color=C_IPOPT, lw=2.5)
    tA, tI = axA.set_title('', fontsize=11), axI.set_title('', fontsize=11)

    axR.set_xscale('symlog', linthresh=1)
    axR.set_yscale('log')
    axR.set_xlim(0, n_adam * 1.1)
    axR.set_ylim(min(ipopt_env.min(), 1e-10) * 0.5, max(adam_kkt[0], ipopt_env[0]) * 3)
    axR.set_xlabel('iteration $k$')
    axR.set_ylabel('best KKT residual so far')
    axR.set_title('distance from optimality', fontsize=11)
    rA, = axR.plot([], [], color=C_ADAM, lw=2, label='Adam')
    rI, = axR.plot([], [], color=C_IPOPT, lw=2, ls='--', label='IPOPT (constrained)')
    mA, = axR.plot([], [], 'o', color=C_ADAM, ms=6)
    mI, = axR.plot([], [], 's', color=C_IPOPT, ms=6)
    axR.legend(fontsize=8, loc='upper right')
    done_txt = axR.text(0.05, 0.06, '', transform=axR.transAxes, fontsize=9,
                        color=C_IPOPT, fontweight='bold')

    def update(f):
        k = frame_its[min(f, len(frame_its) - 1)]
        # Adam panel
        wa = adam_ws[k]
        lnA.set_data(xs.flatten(), forward_numpy(wa, xs, shapes).flatten())
        res_a = adam_kkt[k - 1] if k > 0 else adam_kkt[0]
        tA.set_text(f'Adam — iteration {k}   residual {res_a:.1e}')
        # IPOPT panel (freezes once converged)
        ki = min(k, n_ipopt - 1)
        wi = ipopt_ws[ki]
        lnI.set_data(xs.flatten(), forward_numpy(wi, xs, shapes).flatten())
        if k >= n_ipopt:
            tI.set_text(f'IPOPT — CONVERGED at iteration {n_ipopt - 1}   '
                        f'residual {ipopt_env[-1]:.0e}')
            done_txt.set_text(f'IPOPT done: KKT {ipopt_env[-1]:.0e} '
                              f'at k={n_ipopt - 1} — Adam still going')
        else:
            tI.set_text(f'IPOPT — iteration {ki}   residual {ipopt_env[ki]:.1e}')
        # residual panel
        rA.set_data(np.arange(1, k + 1), adam_kkt[:k])
        if k > 0:
            mA.set_data([k], [adam_kkt[k - 1]])
        rI.set_data(np.arange(min(k, n_ipopt)), ipopt_env[:min(k, n_ipopt)])
        mI.set_data([ki], [ipopt_env[ki]])
        return lnA, lnI, rA, rI, mA, mI, tA, tI, done_txt

    anim = manim.FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.suptitle('The race — same objective, same initial guess $w_0$: '
                 'first-order vs Newton-type convergence')
    fig.tight_layout()
    _save(anim, fig, 'anim_solver_race')


# ─────────────────────────────────────────────────────────────────────────
#  Anim 4 — growing the network: constrained vs unconstrained, per size
# ─────────────────────────────────────────────────────────────────────────

def anim_complexity():
    """Steps H through 4..64 using stored results: left = unconstrained
    Gauss-Newton fit (overfits as capacity grows), middle = constrained IPOPT
    fit (stays smooth — the certificate at work), right = the solve-time race
    building up. All from results/*.json — no re-solving."""
    def load(pat):
        out = {}
        for H in [4, 8, 16, 32, 64]:
            f = f'results/{pat}{H}.json'
            if os.path.exists(f):
                out[H] = json.load(open(f))
        return out

    ipopt = load('exp_size_H')                 # constrained, per H
    gn = load('exp_complexity_gn_H')           # unconstrained baseline, per H
    adam = load('exp_complexity_adam_H')
    Hs = sorted(set(ipopt) & set(gn) & set(adam))
    if not Hs:
        raise SystemExit('missing results — run: python main.py --group size_scaling '
                         'and python main.py --group complexity')

    X_train, y_train, _, _ = generate_dataset(noise_std=0.05, seed=0)
    xs = np.linspace(X_RANGE[0], X_RANGE[1], 300).reshape(1, -1)
    curves_gn = {H: forward_numpy(np.array(gn[H]['w']), xs, param_shapes(1, H, 1)).flatten() for H in Hs}
    curves_ip = {H: forward_numpy(np.array(ipopt[H]['w']), xs, param_shapes(1, H, 1)).flatten() for H in Hs}

    hold = 22                                  # frames per size step
    n_frames = hold * len(Hs) + 12

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=100,
                              gridspec_kw={'width_ratios': [1, 1, 1.15]})
    axG, axI, axT = axes
    pad = 0.25 * (y_train.max() - y_train.min())
    for ax, name in ((axG, 'UNCONSTRAINED (Gauss-Newton/LM)'),
                     (axI, 'CONSTRAINED (IPOPT, Lipschitz + norm-ball)')):
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(y_train.min() - pad, y_train.max() + pad)
        ax.scatter(X_train.flatten(), y_train.flatten(), s=12, c='tab:blue', alpha=0.5)
        ax.set_xlabel('input $x$')
    axG.set_ylabel('output $y$')
    lnG, = axG.plot([], [], color='tab:red', lw=2.5)
    lnI, = axI.plot([], [], color='tab:purple', lw=2.5)
    tG, tI = axG.set_title('', fontsize=10), axI.set_title('', fontsize=10)

    n_vars = {H: ipopt[H]['n_vars'] for H in Hs}
    axT.set_xscale('log'); axT.set_yscale('log')
    axT.set_xlim(min(n_vars.values()) * 0.8, max(n_vars.values()) * 1.3)
    all_t = ([ipopt[H]['solve_time'] for H in Hs] + [gn[H]['solve_time'] for H in Hs]
             + [adam[H]['solve_time'] for H in Hs])
    axT.set_ylim(min(all_t) * 0.5, max(all_t) * 3)
    axT.set_xlabel('decision variables $n$')
    axT.set_ylabel('solve time [s]')
    axT.set_title('the cost of exactness grows — $O(n^3)$', fontsize=10)
    rI, = axT.plot([], [], 'o-', color='tab:blue', label='IPOPT (constrained)')
    rG, = axT.plot([], [], 'd-', color='tab:red', label='GN/LM (unconstrained)')
    rA, = axT.plot([], [], 's-', color='tab:green', label='Adam (unconstrained)')
    axT.legend(fontsize=8, loc='upper left')

    def update(f):
        i = min(f // hold, len(Hs) - 1)
        H = Hs[i]
        lnG.set_data(xs.flatten(), curves_gn[H])
        lnI.set_data(xs.flatten(), curves_ip[H])
        tG.set_text(f'H={H} ({n_vars[H]} weights)   test MSE {gn[H]["test_mse"]:.4f}'
                    f'{"  — overfitting!" if H >= 16 else ""}')
        tI.set_text(f'H={H}   test MSE {ipopt[H]["test_mse"]:.4f}   violation 0')
        sub = Hs[:i + 1]
        rI.set_data([n_vars[h] for h in sub], [ipopt[h]['solve_time'] for h in sub])
        rG.set_data([n_vars[h] for h in sub], [gn[h]['solve_time'] for h in sub])
        rA.set_data([n_vars[h] for h in sub], [adam[h]['solve_time'] for h in sub])
        return lnG, lnI, rI, rG, rA, tG, tI

    anim = manim.FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.suptitle('Growing the network (H = 4 → 64): unconstrained capacity chases noise, '
                 'the constraint keeps its promise')
    fig.tight_layout()
    _save(anim, fig, 'anim_complexity')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['intro', 'sweep', 'race', 'complexity'], default=None)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    if args.only in (None, 'intro'):
        anim_problem_intro()
    if args.only in (None, 'sweep'):
        anim_lipschitz_sweep()
    if args.only in (None, 'race'):
        anim_solver_race()
    if args.only in (None, 'complexity'):
        anim_complexity()
