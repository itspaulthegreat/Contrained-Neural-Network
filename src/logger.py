"""
src/logger.py
Save and load experiment results.
Every run is appended to:
  results/summary.csv     - one row per experiment (scalar metrics only)
  results/<name>.json     - full result for that experiment (incl. weights, history)
"""

import os
import json
import csv
import numpy as np
from datetime import datetime


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


_CSV_FIELDS = [
    'timestamp', 'name', 'label', 'group', 'method', 'success',
    'H', 'n_vars', 'n_constraints', 'n_iter', 'solve_time',
    'train_mse', 'test_mse', 'lipschitz_estimate', 'max_constraint_violation',
    'hessian_condition_number', 'L_max', 'B_max', 'noise_std',
]


def save_result(res: dict, results_dir: str = 'results'):
    """Append scalar metrics to summary.csv and dump the full result as JSON."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    csv_path = os.path.join(results_dir, 'summary.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        row = {k: res.get(k, '') for k in _CSV_FIELDS}
        row['timestamp'] = ts
        writer.writerow(row)

    json_path = os.path.join(results_dir, f'{res["name"]}.json')
    payload = dict(res)
    payload['timestamp'] = ts
    with open(json_path, 'w') as f:
        json.dump(payload, f, cls=_NpEncoder, indent=2)


def load_result(name: str, results_dir: str = 'results') -> dict:
    """Load one experiment's full JSON result by name."""
    path = os.path.join(results_dir, f'{name}.json')
    with open(path) as f:
        return json.load(f)


def load_summary(results_dir: str = 'results') -> list:
    """Load summary.csv as a list of row dicts (empty if it does not exist)."""
    path = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def print_summary(results: list):
    """Pretty-print a table of experiment results to stdout."""
    cols = ['name', 'method', 'train_mse', 'test_mse', 'n_iter', 'solve_time']
    widths = [28, 8, 12, 12, 8, 12]

    header = '  '.join(f'{c:<{w}}' for c, w in zip(cols, widths))
    sep = '  '.join('-' * w for w in widths)
    print('\n' + sep)
    print(header)
    print(sep)

    for r in results:
        row = [
            str(r.get('name', ''))[:28],
            str(r.get('method', ''))[:8],
            f'{float(r.get("train_mse", 0)):.5f}',
            f'{float(r.get("test_mse", 0)):.5f}',
            str(r.get('n_iter', '')),
            f'{float(r.get("solve_time", 0)):.3f} s',
        ]
        print('  '.join(f'{v:<{w}}' for v, w in zip(row, widths)))
    print(sep + '\n')
