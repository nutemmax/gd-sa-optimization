# benchmark_experiments_best.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks import rosenbrock, rastrigin, ackley, grad_rosenbrock, grad_rastrigin, grad_ackley
from gd import gradient_descent
from sa import sa_continuous
from utils import bootstrap_experiment, generate_summary_csv, get_experiment_id

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')
plots_dir = os.path.join(results_dir, 'plots')
analytical_dir = os.path.join(results_dir, 'analytical')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

with open(os.path.join(results_dir, 'best_hyperparams.json')) as f:
    best_params = json.load(f)

benchmarks = {
    'rosenbrock': (rosenbrock, grad_rosenbrock),
    'rastrigin': (rastrigin, grad_rastrigin),
    'ackley': (ackley, grad_ackley)
}

init_point = None
bounds = [(-5, 5), (-5, 5)]
tol = 1e-6
perturbation_method = 'normal'
adaptive_step_size = False
num_runs = 50
experiment_id = get_experiment_id()


def plot_best_convergence(name, sa_histories, gd_histories):
    sa_best_idx = np.argmin([hist[-1] for hist in sa_histories])
    gd_best_idx = np.argmin([hist[-1] for hist in gd_histories])

    plt.figure(figsize=(10, 6))
    plt.plot(sa_histories[sa_best_idx], label='SA (Best Run)')
    plt.plot(gd_histories[gd_best_idx], label='GD (Best Run)')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title(f'Best-run Convergence - {name} (Best Params)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{name}_best_hyperparameters_convergence_exp{experiment_id}.png'))
    plt.close()


def run_experiments():
    for name, (f, grad) in benchmarks.items():
        print(f'Running {name}')

        sa_p = best_params[name]['sa']
        gd_p = best_params[name]['gd']

        sa_results = bootstrap_experiment(sa_continuous, num_runs, f, x_init=init_point, bounds=bounds, 
                                          T0=sa_p['T0'], alpha=sa_p['alpha'], step_size=sa_p['step_size'],
                                          tol=tol, max_iter=20000)

        gd_results = bootstrap_experiment(gradient_descent, num_runs, f, grad, 
                                          lr=gd_p['lr'], tol=tol, max_iter=20000, x_init=init_point)

        generate_summary_csv(f'{name}_best_hyperparameters', gd_stats=gd_results['stats'], 
                             sa_stats=sa_results['stats'], experiment_id=experiment_id, save_dir=analytical_dir)

        plot_best_convergence(name, sa_results['histories'], gd_results['histories'])


if __name__ == '__main__':
    run_experiments()
    print('All benchmark best-params experiments complete.')