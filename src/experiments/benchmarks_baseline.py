import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt

from src.problems.benchmarks import rosenbrock, rastrigin, ackley, grad_rosenbrock, grad_rastrigin, grad_ackley
from src.optimizers.sa import sa_continuous
from src.optimizers.gd import gradient_descent
from src.utils.utils_experiments import bootstrap_experiment_benchmarks, get_experiment_id, generate_summary_csv

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
tol = 1e-6
perturbation_method = "normal"
adaptive_step_size = False
num_runs = 50
dim = 2
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


def plot_best_convergence(name, sa_histories, gd_histories, f):
    sa_final_vals = [f(hist[-1]) for hist in sa_histories]
    gd_final_vals = [f(hist[-1]) for hist in gd_histories]

    sa_best_idx = np.argmin(sa_final_vals)
    gd_best_idx = np.argmin(gd_final_vals)

    plt.figure(figsize=(10, 6))
    plt.plot([f(x) for x in sa_histories[sa_best_idx]], label="SA (Best Run)")
    plt.plot([f(x) for x in gd_histories[gd_best_idx]], label="GD (Best Run)")
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

        # select init range based on the benchmark
        if name.lower() == "rosenbrock":
            init_range = (-2, 2)
        else:
            init_range = (-5, 5)

        # pre-generate the same initializations for all runs
        inits = [np.random.uniform(*init_range, size=dim) for _ in range(num_runs)]

        # === GD ===
        gd_results = bootstrap_experiment_benchmarks(
            algorithm_function=gradient_descent,
            runs=num_runs,
            f=f,
            grad_f=grad,
            dim=dim,
            x_inits = inits,
            name=name,
            f_star=0.0,
            x_star=np.zeros(2),
            **gd_params
        )

        # === SA ===
        sa_results = bootstrap_experiment_benchmarks(
            algorithm_function=sa_continuous,
            runs=num_runs,
            f=f,
            dim=2,
            x_inits = inits,
            name=name,
            f_star=0.0,
            x_star=np.zeros(2),
            tol=tol,
            perturbation_method=perturbation_method,
            adaptive_step_size=adaptive_step_size,
            **sa_params
        )

        generate_summary_csv(f"{name}_baseline_parameters", gd_results['stats'], sa_results['stats'], experiment_id, analytical_dir)
        plot_best_convergence(name, sa_results['histories'], gd_results['histories'], f)


if __name__ == "__main__":
    run_experiments()
    print("All baseline benchmark experiments complete.")
