"""
mse_table.py
──────────────
The predictive-performance companion to the sensitivity table: median test
MSE, inter-quartile range, full range, and head-to-head win rate for IPOPT
(hard constraint) vs the best tuned AdamW, across N seeds and each sigma.

Together with sensitivity_study.py this closes the loop between prediction
performance and constraint satisfaction -- the two axes a viva asks about.

    python mse_table.py     # writes results/mse_table.json + prints the table
"""

import json
import os

import numpy as np

from seed_study import run_ipopt, run_adam, tune_wd, N_SEEDS

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def main(sigmas=(0.1, 0.3)):
    table = {}
    for sigma in sigmas:
        wd = tune_wd(sigma)
        ip = np.array([run_ipopt(sigma, s)[0] for s in range(N_SEEDS)])
        aw = np.array([run_adam(sigma, s, wd)[0] for s in range(N_SEEDS)])
        ipopt_wins = int(np.sum(ip < aw))          # paired, per seed
        adam_wins = N_SEEDS - ipopt_wins
        table[sigma] = dict(wd=wd, n=N_SEEDS)
        for name, a, wins in [('IPOPT (hard constraint)', ip, ipopt_wins),
                              (f'AdamW (wd={wd:g})', aw, adam_wins)]:
            table[sigma][name] = dict(
                median=float(np.median(a)), mean=float(a.mean()),
                q1=float(np.percentile(a, 25)), q3=float(np.percentile(a, 75)),
                min=float(a.min()), max=float(a.max()),
                win_rate=f'{wins}/{N_SEEDS}',
                values=sorted(round(float(x), 5) for x in a))

    with open(os.path.join(RESULTS, 'mse_table.json'), 'w') as f:
        json.dump(table, f, indent=2)

    print('=' * 92)
    print(f"{'sigma':>6} {'method':<24} {'median MSE':>11} {'IQR':>17} "
          f"{'range':>17} {'win rate':>9}")
    print('-' * 92)
    for sigma, d in table.items():
        for name in [k for k in d if k not in ('wd', 'n')]:
            s = d[name]
            print(f"{sigma:>6} {name:<24} {s['median']:>11.4f} "
                  f"[{s['q1']:.4f},{s['q3']:.4f}] [{s['min']:.4f},{s['max']:.4f}] "
                  f"{s['win_rate']:>9}")
        print('-' * 92)
    print('win rate = number of seeds (out of 15) on which that method had the lower test MSE')


if __name__ == '__main__':
    main()
