import os
import json
import pandas as pd

# === Setup ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GRIDSEARCH_DIR = os.path.join(ROOT_DIR, 'results', 'gridsearch')
OUTPUT_PATH = os.path.join(GRIDSEARCH_DIR, 'best_hyperparams.json')

benchmarks = ['rosenbrock', 'rastrigin', 'ackley']
ising_variants = ['ising_relaxed', 'ising_discrete']

# Maps our canonical names to actual filenames
FILENAME_MAP = {
    'rosenbrock': {
        'sa': 'gridsearch_sa_rosenbrock.csv',
        'gd': 'gridsearch_gd_rosenbrock.csv'
    },
    'rastrigin': {
        'sa': 'gridsearch_sa_rastrigin.csv',
        'gd': 'gridsearch_gd_rastrigin.csv'
    },
    'ackley': {
        'sa': 'gridsearch_sa_ackley.csv',
        'gd': 'gridsearch_gd_ackley.csv'
    },
    'ising_relaxed': {
        'sa': 'gridsearch_sa_continuous_ising.csv',
        'gd': 'gridsearch_gd_ising.csv'
    },
    'ising_discrete': {
        'sa': 'gridsearch_sa_discrete_ising.csv'
    }
}

def load_best_config(path):
    """Loads CSV and returns best row by lowest mean value."""
    df = pd.read_csv(path)
    if 'mean' not in df.columns:
        raise ValueError(f"No 'mean' column found in {path}")
    best_row = df.loc[df['mean'].idxmin()]
    return best_row.to_dict()

# === Select best hyperparams ===
best_params = {}

# --- Benchmarks ---
for benchmark in benchmarks:
    best_params[benchmark] = {}
    for algo in ['sa', 'gd']:
        filename = FILENAME_MAP[benchmark][algo]
        path = os.path.join(GRIDSEARCH_DIR, filename)
        if os.path.exists(path):
            best_params[benchmark][algo] = load_best_config(path)
        else:
            print(f"Missing: {path}")
            best_params[benchmark][algo] = "Gridsearch not found"

# --- Ising variants ---
best_params['ising'] = {}
for variant in ising_variants:
    best_params['ising'][variant] = {}
    for algo in FILENAME_MAP[variant]:
        filename = FILENAME_MAP[variant][algo]
        path = os.path.join(GRIDSEARCH_DIR, filename)
        if os.path.exists(path):
            best_params['ising'][variant][algo] = load_best_config(path)
        else:
            print(f"Missing: {path}")
            best_params['ising'][variant][algo] = "Gridsearch not found"

# === Save JSON ===
with open(OUTPUT_PATH, 'w') as f:
    json.dump(best_params, f, indent=4)
print(f"Saved best hyperparameters to: {OUTPUT_PATH}")
