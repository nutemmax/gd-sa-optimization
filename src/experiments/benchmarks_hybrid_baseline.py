# src/experiments/benchmarks_hybrid.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.problems.benchmarks import (
    rosenbrock, rastrigin, ackley,
    grad_rosenbrock, grad_rastrigin, grad_ackley
)
from src.optimizers.hybrid import sa_gd_hybrid
from src.utils.utils_experiments import bootstrap_experiment_benchmarks, get_experiment_id, generate_summary_csv
from src.utils.utils_plots import plot_energy_trajectory, plot_final_spin_config

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === Config ===
benchmark_funcs = {
    "rosenbrock": (rosenbrock, grad_rosenbrock, 0.0, np.array([1.0, 1.0])),
    "rastrigin": (rastrigin, grad_rastrigin, 0.0, np.array([0.0, 0.0])),
    "ackley": (ackley, grad_ackley, 0.0, np.array([0.0, 0.0]))
}

baseline_params = {
    "lr": 0.1,
    "sigma": 0.1,
    "T0": 0.1,
    "max_iter": 50000,
    "tol": 1e-6
}

num_runs = 50
exp_id = get_experiment_id()
dim=2
ascent_method = "ascent"

def plot_best_hybrid_convergence(name, f_histories, f):
    final_vals = [hist[-1] for hist in f_histories]
    best_idx = np.argmin(final_vals)
    best_history = f_histories[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label="Hybrid SA-GD (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(f"Best-run Convergence on {name} (Hybrid Baseline)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_hybrid-{ascent_method}_baseline_convergence_exp{exp_id}.png"))
    plt.close()


# === Run Hybrid SA-GD ===
for name, (f, grad_f, f_star, x_star) in benchmark_funcs.items():
    print(f"\n==== Running Hybrid SA-GD on {name.upper()} ====")

    # select init range based on the benchmark
    if name.lower() == "rosenbrock":
        init_range = (-2, 2)
        x_star = np.array([1.0, 1.0])
    else:
        init_range = (-5, 5)
        x_star = np.zeros(2)

    print(f"Function : {name}, x_star : {x_star}")

    # pre-generate the same initializations for all runs
    inits = [np.random.uniform(*init_range, size=dim) for _ in range(num_runs)]

    output = bootstrap_experiment_benchmarks(
        algorithm_function=sa_gd_hybrid,
        runs=num_runs,
        f=f,
        grad_f=grad_f,
        f_star=f_star,
        x_star=x_star,
        name=name,
        x_inits=inits,
        ascent_method = ascent_method,
        init_range = init_range,
        **baseline_params
    )

    # Save CSV summary
    generate_summary_csv(
        f"{name}_hybrid-{ascent_method}_baseline_parameters",
        sa_stats=None,
        gd_stats=None,
        hybrid_stats=output['stats'],
        experiment_id=exp_id,
        save_dir=analytical_dir,
        name_alg = f"SA-GD-{ascent_method}"
    )

    # Plot best convergence
    plot_best_hybrid_convergence(
        name=name,
        f_histories=output["f_histories"],
        f=f
    )

    print(f"Saved summary and plot for {name} (Experiment ID {exp_id})")
