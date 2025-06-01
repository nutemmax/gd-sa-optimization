# benchmark_experiments_best.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.problems.benchmarks import rosenbrock, rastrigin, ackley, grad_rosenbrock, grad_rastrigin, grad_ackley
from src.optimizers.gd import gradient_descent
from src.optimizers.sa import sa_continuous
from src.utils.utils_experiments import bootstrap_experiment_benchmarks, generate_summary_csv, get_experiment_id

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, '..', 'results')
plots_dir = os.path.join(results_dir, 'plots')
analytical_dir = os.path.join(results_dir, 'analytical')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

with open(os.path.join(results_dir, 'gridsearch','best_hyperparams.json')) as f:
    best_params = json.load(f)

benchmarks = {
    'rosenbrock': (rosenbrock, grad_rosenbrock),
    'rastrigin': (rastrigin, grad_rastrigin),
    'ackley': (ackley, grad_ackley)
}

init_point = None
tol = 1e-6
perturbation_method = 'normal'
adaptive_step_size = False
num_runs = 5
experiment_id = get_experiment_id()
dim = 2


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
    plt.title(f"Best-run Convergence on {name} (Best Params)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_best_parameters_convergence_exp{experiment_id}.png"))
    plt.close()


def run_experiments():
    for name, (f, grad) in benchmarks.items():
        print(f'Running {name}')

        # select init range based on the benchmark
        if name.lower() == "rosenbrock":
            init_range = (-2, 2)
        else:
            init_range = (-5, 5)

        sa_p = best_params[name]['sa']
        gd_p = best_params[name]['gd']

        inits = [np.random.uniform(*init_range, size=dim) for _ in range(num_runs)]

        # === GD ====
        gd_results = bootstrap_experiment_benchmarks(
            algorithm_function=gradient_descent,
            runs=num_runs,
            f=f,
            grad_f=grad,
            dim=dim,
            x_inits=inits,
            name=name,
            f_star=0.0,
            x_star=np.zeros(2),
            tol=tol,
            lr=gd_p['lr'],
            max_iter=20000,
        )

        # === SA ====
        sa_results = bootstrap_experiment_benchmarks(
            algorithm_function=sa_continuous,
            runs=num_runs,
            f=f,
            dim=dim,
            x_inits=inits,
            name=name,
            f_star=0.0,
            x_star=np.zeros(2),
            tol=tol,
            perturbation_method=perturbation_method,
            adaptive_step_size=adaptive_step_size,
            T0=sa_p['T0'],
            alpha=sa_p['alpha'],
            step_size=sa_p['step_size'],
            max_iter=20000
        )

        generate_summary_csv(f'{name}_best_hyperparameters', gd_stats=gd_results['stats'], 
                             sa_stats=sa_results['stats'], experiment_id=experiment_id, save_dir=analytical_dir)

        plot_best_convergence(name, sa_results['histories'], gd_results['histories'], f)


if __name__ == '__main__':
    run_experiments()
    print('All benchmark best-params experiments complete.')