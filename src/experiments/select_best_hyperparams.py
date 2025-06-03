import os
import json
import pandas as pd

# === Setup ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FINAL_DIR = os.path.join(ROOT_DIR, 'results', 'gridsearch', 'final')
HYBRID_DIR = os.path.join(ROOT_DIR, 'results', 'gridsearch')
OUTPUT_PATH = os.path.join(FINAL_DIR, 'best_hyperparams.json')

benchmarks = ['rosenbrock', 'rastrigin', 'ackley']
ising_variants = ['ising_relaxed', 'ising_discrete']

FILENAME_MAP = {
    'rosenbrock': {
        'sa': 'gridsearch_sa_rosenbrock_new.csv',
        'gd': 'gridsearch_gd_rosenbrock_new.csv',
        'hybrid': 'gridsearch_hybrid-unif_rosenbrock_new.csv'
    },
    'rastrigin': {
        'sa': 'gridsearch_sa_rastrigin_new.csv',
        'gd': 'gridsearch_gd_rastrigin_new.csv',
        'hybrid': 'gridsearch_hybrid-unif_rastrigin_new.csv'
    },
    'ackley': {
        'sa': 'gridsearch_sa_ackley_new.csv',
        'gd': 'gridsearch_gd_ackley_new.csv',
        'hybrid': 'gridsearch_hybrid-unif_ackley_new.csv'
    },
    'ising_relaxed': {
        'sa': 'gridsearch_sa_continuous_ising_new.csv',
        'gd': 'gridsearch_gd_continuous_ising_new.csv'
    },
    'ising_discrete': {
        'sa': 'gridsearch_sa_discrete_ising_new.csv'
    }
}

def load_best_config(path, select_by="rmse"):
    df = pd.read_csv(path)
    if select_by not in df.columns:
        raise ValueError(f"No '{select_by}' column found in {path}")
    return df.loc[df[select_by].idxmin()].to_dict()

# === Select best hyperparams ===
best_params = {}

# --- Benchmarks ---
for benchmark in benchmarks:
    best_params[benchmark] = {}
    for algo in ['sa', 'gd', 'hybrid']:
        filename = FILENAME_MAP[benchmark].get(algo)
        if filename:
            dir_path = HYBRID_DIR if algo == 'hybrid' else FINAL_DIR
            path = os.path.join(dir_path, filename)
            if os.path.exists(path):
                metric = "rmed" if algo == "hybrid" else "rmse"
                best_params[benchmark][algo] = load_best_config(path, select_by=metric)
            else:
                print(f"Missing: {path}")
                best_params[benchmark][algo] = "Gridsearch not found"

# --- Ising variants ---
best_params['ising'] = {}
for variant in ising_variants:
    best_params['ising'][variant] = {}
    for algo in FILENAME_MAP[variant]:
        filename = FILENAME_MAP[variant][algo]
        path = os.path.join(FINAL_DIR, filename)
        if os.path.exists(path):
            best_params['ising'][variant][algo] = load_best_config(path)
        else:
            print(f"Missing: {path}")
            best_params['ising'][variant][algo] = "Gridsearch not found"

# === Save JSON ===
os.makedirs(FINAL_DIR, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Saved best hyperparameters to: {OUTPUT_PATH}")
