# benchmark_experiments_baseline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks import rosenbrock, rastrigin, ackley, grad_rosenbrock, grad_rastrigin, grad_ackley
from sa import sa_continuous
from gd import gradient_descent
from utils import bootstrap_experiment, get_experiment_id, generate_summary_csv

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
analytical_dir = os.path.join(base_dir, "results", "analytical")
plots_dir = os.path.join(base_dir, "results", "plots")
os.makedirs(analytical_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

benchmarks = {
    "rosenbrock": (rosenbrock, grad_rosenbrock),
    "rastrigin": (rastrigin, grad_rastrigin),
    "ackley": (ackley, grad_ackley)
}

init_point = None
bounds = [(-5, 5), (-5, 5)]
tol = 1e-6
perturbation_method = "normal"
adaptive_step_size = False
num_runs = 50
experiment_id = get_experiment_id()

sa_params = {
    "T0": 50.0,
    "alpha": 0.99,
    "step_size": 0.1,
    "max_iter": 20000
}

gd_params = {
    "lr": 0.01,
    "max_iter": 20000
}


def plot_best_convergence(name, sa_histories, gd_histories):
    sa_best_idx = np.argmin([hist[-1] for hist in sa_histories])
    gd_best_idx = np.argmin([hist[-1] for hist in gd_histories])

    plt.figure(figsize=(10, 6))
    plt.plot(sa_histories[sa_best_idx], label="SA (Best Run)")
    plt.plot(gd_histories[gd_best_idx], label="GD (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(f"Best-run Convergence on {name} (Baseline Params)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_baseline_parameters_convergence_exp{experiment_id}.png"))
    plt.close()


def run_experiments():
    for name, (f, grad) in benchmarks.items():
        print(f"\nRunning experiments on {name}")

        gd_results = bootstrap_experiment(gradient_descent, num_runs, f, grad, **gd_params, tol=tol, x_init=init_point)
        sa_results = bootstrap_experiment(sa_continuous, num_runs, f, **sa_params, tol=tol, x_init=init_point, bounds=bounds, perturbation_method=perturbation_method, adaptive_step_size=adaptive_step_size)

        generate_summary_csv(f"{name}_baseline_parameters", gd_results['stats'], sa_results['stats'], experiment_id, analytical_dir)
        plot_best_convergence(name, sa_results['histories'], gd_results['histories'])


if __name__ == "__main__":
    run_experiments()
    print("All baseline benchmark experiments complete.")
