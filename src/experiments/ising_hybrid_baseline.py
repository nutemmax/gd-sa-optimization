# src/experiments/ising_hybrid_baseline.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.optimizers.hybrid import sa_gd_hybrid
from src.problems.ising import relaxed_ising_energy, grad_relaxed_ising
from src.utils.utils_experiments import bootstrap_experiment_ising, get_experiment_id, generate_summary_csv
from src.utils.utils_plots import plot_energy_trajectory

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === Shared settings ===
num_runs = 5
tol = 1e-6
lattice_shape = (10, 10)
experiment_id = get_experiment_id()
dim = lattice_shape[0]

baseline_params = {"lr": 0.1, "sigma": 0.1, "T0": 0.1, "max_iter": 20000, "tol": tol}

relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()

x_inits = [np.random.uniform(-1, 1, size=lattice_shape) for _ in range(num_runs)]
x_inits_flat = [x.flatten() for x in x_inits]

for ascent_method in ["ascent", "unif"]:
    print(f"\nRunning Hybrid SA-GD (baseline) with ascent_method = {ascent_method}")

    results = bootstrap_experiment_ising(
        algorithm_function=sa_gd_hybrid,
        runs=num_runs,
        f=relaxed_f,
        grad_f=grad_wrapped,
        dim=dim,
        x_inits=x_inits_flat,
        f_star=-200.0,
        name="ising_relaxed",
        ascent_method=ascent_method,
        **baseline_params
    )

    generate_summary_csv(
        benchmark_name=f"ising_relaxed_hybrid-{ascent_method}_baseline_parameters",
        gd_stats=None,
        sa_stats=None,
        hybrid_stats=results["stats"],
        experiment_id=experiment_id,
        save_dir=analytical_dir,
        name_alg=f"SA-GD-{ascent_method}"
    )

    best_run = min(results["histories"], key=lambda h: relaxed_f(h[-1]))
    energy_values = [relaxed_f(x) for x in best_run]

    plt.figure(figsize=(10, 6))
    plt.plot(energy_values, label=f"Hybrid ({ascent_method}) Best Run")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Relaxed Ising: Hybrid ({ascent_method}) Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"ising_relaxed_hybrid-{ascent_method}_baseline_convergence_exp{experiment_id}.png"
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

print("\nAll hybrid baseline experiments on relaxed Ising complete.")
