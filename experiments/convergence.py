"""Convergence diagnostic: IPOPT KKT residual and Adam gradient norm vs
iteration on the constrained problem -> results/convergence.json. The two are
solver-specific quantities, compared only qualitatively (see the report)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiments import _make
from src.data import generate_dataset
from src.convergence import solve_with_iterates
from src.model import param_shapes, random_init
from src.baseline_adam import adam_optimize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, B = 4.0, 6.0

Xtr, ytr, Xte, yte = generate_dataset(n_train=60, n_test=40, noise_std=0.05, seed=0)


def main():
    """Record IPOPT KKT residuals and Adam gradient norms -> convergence.json."""
    exp = _make("conv_ipopt", "convergence", "convergence", method="ipopt", H=8,
                use_lipschitz=True, use_norm_ball=True, L_max=L, B_max=B, noise_std=0.05,
                ipopt_opts=dict(ipopt=dict(max_iter=3000, tol=1e-9, print_level=0),
                                print_time=False))
    r = solve_with_iterates(exp, Xtr, ytr, Xte, yte)
    kkt = r["kkt_history"]

    shapes = param_shapes(1, 8, 1)
    out = adam_optimize(random_init(shapes, seed=0), shapes, Xtr, ytr, lr=0.02,
                        n_iter=40000, weight_decay=0.0, tol=1e-9)
    grad = out["grad_inf_history"]

    result = dict(ipopt_kkt=kkt, ipopt_iters=len(kkt),
                  adam_grad=grad, adam_iters=len(grad))
    path = os.path.join(HERE, "results", "convergence.json")
    json.dump(result, open(path, "w"), indent=1, default=float)
    print(f"IPOPT: {len(kkt)} iters, final KKT residual {kkt[-1]:.2e}")
    print(f"Adam:  {len(grad)} iters, final gradient norm {grad[-1]:.2e}")
    print("wrote", path)


if __name__ == "__main__":
    main()
