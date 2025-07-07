import os
import json
import pandas as pd

# === setup ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FINAL_DIR = os.path.join(ROOT_DIR, 'results', 'gridsearch', 'final')
OUTPUT_PATH = os.path.join(FINAL_DIR, 'best_hyperparams.json')

benchmarks = ['rosenbrock', 'rastrigin', 'ackley']
ising_variants = ['ising_relaxed', 'ising_discrete']

# === hybrid ascent variants ===
hybrid_methods = ['unif', 'ascent']  # 'unif' = random perturbation (SAGDP), 'ascent' = gradient ascent (SAGDA)

def load_best_config(path, select_by="rmse"):
    df = pd.read_csv(path)
    if select_by not in df.columns:
        raise ValueError(f"No '{select_by}' column found in {path}")
    return df.loc[df[select_by].idxmin()].to_dict()

# === select best hyperparams ===
best_params = {}

# benchmarks 
for benchmark in benchmarks:
    best_params[benchmark] = {}

    # SA and GD
    for algo in ['sa', 'gd']:
        filename = f"gridsearch_{algo}_{benchmark}_new.csv"
        path = os.path.join(FINAL_DIR, filename)
        if os.path.exists(path):
            best_params[benchmark][algo] = load_best_config(path, select_by="rmed")
        else:
            print(f"Missing: {path}")
            best_params[benchmark][algo] = "Gridsearch not found"

    # hybrid variants
    for method in hybrid_methods:
        key = f"hybrid-{method}"
        filename = f"gridsearch_hybrid-{method}_{benchmark}_new.csv"
        path = os.path.join(FINAL_DIR, filename)
        if os.path.exists(path):
            best_params[benchmark][key] = load_best_config(path, select_by="rmed")
        else:
            print(f"Missing: {path}")
            best_params[benchmark][key] = "Gridsearch not found"

# Ising variants
best_params['ising'] = {}
for variant in ising_variants:
    best_params['ising'][variant] = {}

    # SA
    sa_filename = f"gridsearch_sa_{'discrete' if variant == 'ising_discrete' else 'continuous'}_ising_new.csv"
    sa_path = os.path.join(FINAL_DIR, sa_filename)
    if os.path.exists(sa_path):
        best_params['ising'][variant]['sa'] = load_best_config(sa_path, select_by="rmse")
    else:
        print(f"Missing: {sa_path}")
        best_params['ising'][variant]['sa'] = "Gridsearch not found"

    # GD (only for relaxed version)
    if variant == 'ising_relaxed':
        gd_filename = f"gridsearch_gd_continuous_ising_new.csv"
        gd_path = os.path.join(FINAL_DIR, gd_filename)
        if os.path.exists(gd_path):
            best_params['ising'][variant]['gd'] = load_best_config(gd_path, select_by="rmse")
        else:
            print(f"Missing: {gd_path}")
            best_params['ising'][variant]['gd'] = "Gridsearch not found"

        # hybrid variants
        for method in hybrid_methods:
            key = f"hybrid-{method}"
            hybrid_filename = f"gridsearch_hybrid-{method}_ising_relaxed_new.csv"
            hybrid_path = os.path.join(FINAL_DIR, hybrid_filename)
            if os.path.exists(hybrid_path):
                best_params['ising'][variant][key] = load_best_config(hybrid_path, select_by="rmed")
            else:
                print(f"Missing: {hybrid_path}")
                best_params['ising'][variant][key] = "Gridsearch not found"

# === save JSON ===
os.makedirs(FINAL_DIR, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Saved best hyperparameters to: {OUTPUT_PATH}")
