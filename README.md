# Neural Network Training as a Constrained Nonlinear Program

Training a small neural network, written as a constrained nonlinear program
(NLP) and solved with an interior-point method, and compared against the
standard machine-learning approach of soft penalties trained by Adam.

## Idea

The weights of a one-hidden-layer `tanh` network are the decision variables of
an NLP whose objective is the mean-squared training error and whose constraints
encode properties the trained model must satisfy:

- **Lipschitz (sensitivity) bound** `‖w1‖² ‖w2‖² ≤ L²` - bounds how fast the
  output can change with the input (bilinear, nonconvex).
- **Norm ball** `‖w‖² ≤ B²` - a hard version of L2 regularization (convex).
- **Bias ordering** `b1[0] ≤ … ≤ b1[H-1]` - removes the permutation symmetry of
  the hidden units (linear).

The NLP is built symbolically with [CasADi](https://web.casadi.org/) and solved
with [IPOPT](https://github.com/coin-or/Ipopt). The interior-point solve
satisfies the constraints to solver tolerance; the penalty baselines (Adam,
AdamW, penalty-Adam) only approach this behaviour under conditions that are easy
to get wrong. The network is a convenient nonconvex function on which to study
the optimization problem, not the point of the project.

## Layout

```
nn_constrained_nlp/
├── main.py                 CLI that runs the experiment groups in configs/
├── configs/experiments.py  every experiment definition (the only file to edit)
├── src/                    the library
│   ├── model.py            network forward pass and weight (un)flattening
│   ├── deep_model.py       general L-hidden-layer network (used by the depth study)
│   ├── data.py             synthetic teacher-student and pendulum datasets
│   ├── constraints.py      Lipschitz / norm-ball / spectral / ordering builders
│   ├── nlp_builder.py      assembles the CasADi NLP from an experiment config
│   ├── solver.py           dispatches to IPOPT / SQP / Adam, unified result dict
│   ├── baseline_adam.py    Adam / AdamW baseline
│   ├── penalty_adam.py     hinge-penalty baseline
│   ├── analysis.py         sensitivity estimate, violation, conditioning
│   ├── callbacks.py        IPOPT per-iteration callback
│   └── ...                 kkt, multistart, warm_start, convergence, plotting
├── experiments/            standalone studies that build on the library
│   ├── synthetic_protocol.py   three-way-split study on synthetic data
│   ├── pendulum_protocol.py    validation-based pendulum system ID
│   ├── pendulum_convergence.py hard vs soft, run to convergence
│   ├── exact_penalty.py        exact-penalty threshold sweep
│   ├── sensitivity.py          achieved sensitivity across seeds
│   ├── seed_study.py           seed-averaged accuracy comparison
│   ├── scaling_study.py        matched IPOPT vs penalty-Adam as width grows
│   ├── depth_study.py          matched IPOPT vs penalty-Adam as depth grows
│   ├── symmetry_ablation.py    ablation of the bias-ordering constraint
│   └── report_figures.py       regenerates the report figures
├── tests/                  unit tests (model, constraints, warm start)
├── results/                generated JSON results
└── figures/                generated figures
```

## Install

```bash
pip install -r requirements.txt
```

## Run

The experiment harness runs the groups defined in `configs/experiments.py`:

```bash
python main.py --dry-run                 # list the experiments
python main.py --group size_scaling      # one group
python main.py --group complexity
python main.py --name exp_method_ipopt    # a single experiment
```

Results are written to `results/` and figures to `figures/`. The standalone
studies are run as modules from the repository root, for example:

```bash
python -m experiments.pendulum_convergence
python -m experiments.exact_penalty
python -m experiments.scaling_study
python -m experiments.depth_study
python -m experiments.symmetry_ablation
python -m experiments.sensitivity
```

After the studies have produced their result files, regenerate the report
figures with:

```bash
python -m experiments.report_figures
```

## Tests

```bash
python -m pytest tests/
```
