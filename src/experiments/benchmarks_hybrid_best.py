# src/experiments/benchmarks_hybrid_best.py

import sys
import os
import json
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
from src.utils.utils_plots import plot_energy_trajectory

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
gridsearch_dir = os.path.join(results_dir, "gridsearch")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === Load best hybrid hyperparameters ===
with open(os.path.join(gridsearch_dir, "best_hyperparams.json"), "r") as f:
    best_params = json.load(f)

# === Config ===
benchmark_funcs = {
    "rosenbrock": (rosenbrock, grad_rosenbrock, 0.0, np.array([1.0, 1.0])),
    "rastrigin": (rastrigin, grad_rastrigin, 0.0, np.array([0.0, 0.0])),
    "ackley": (ackley, grad_ackley, 0.0, np.array([0.0, 0.0]))
}

num_runs = 50
exp_id = get_experiment_id()
dim = 2
ascent_method = "ascent"

def plot_best_hybrid_convergence(name, f_histories, f):
    final_vals = [hist[-1] for hist in f_histories]
    best_idx = np.argmin(final_vals)
    best_history = f_histories[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label="Hybrid SA-GD (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(f"Best-run Convergence on {name} (Hybrid Best)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_hybrid-{ascent_method}_best_convergence_exp{exp_id}.png"))
    plt.close()

# === Run Hybrid SA-GD with Best Params ===
for name, (f, grad_f, f_star, x_star) in benchmark_funcs.items():
    print(f"\n==== Running Hybrid SA-GD on {name.upper()} ====")

    # extract best hyperparams for hybrid
    hybrid_params = best_params[name][f"hybrid-{ascent_method}"]
    lr = hybrid_params["lr"]
    sigma = hybrid_params["sigma"]
    T0 = hybrid_params["T0"]

    # select init range based on the benchmark
    if name.lower() == "rosenbrock":
        init_range = (-2, 2)
        x_star = np.array([1.0, 1.0])
    else:
        init_range = (-5, 5)
        x_star = np.zeros(2)

    print(f"Function : {name}, x_star : {x_star}")

    # shared inits for all runs
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
        lr=lr,
        sigma=sigma,
        T0=T0,
        max_iter=50000,
        tol=1e-6,
        ascent_method = ascent_method,
        init_range=init_range
    )

    # save summary
    generate_summary_csv(
        f"{name}_hybrid-{ascent_method}_best_parameters",
        sa_stats=None,
        gd_stats=None,
        hybrid_stats=output['stats'],
        experiment_id=exp_id,
        save_dir=analytical_dir,
        name_alg=f"SA-GA-{ascent_method}"
    )

    # best convergence plot
    plot_best_hybrid_convergence(name=name, f_histories=output["f_histories"], f=f)

    print(f"Saved summary and plot for {name} (Experiment ID {exp_id})")

