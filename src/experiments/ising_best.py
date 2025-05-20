# src/experiments/ising_best.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.optimizers.sa import sa_continuous, sa_discrete
from src.optimizers.gd import gradient_descent
from src.problems.ising import ising_energy, relaxed_ising_energy, grad_relaxed_ising
from src.utils.utils_experiments import bootstrap_experiment, get_experiment_id

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
gridsearch_dir = os.path.join(results_dir, "gridsearch")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

with open(os.path.join(gridsearch_dir, "best_hyperparams.json")) as f:
    best_params = json.load(f)

# === Shared settings ===
num_runs = 50
tol = 1e-6
lattice_shape = (10, 10)
experiment_id = get_experiment_id()

def plot_combined_convergence(name, sa_histories, gd_histories):
    sa_best = min(sa_histories, key=lambda h: h[-1])
    gd_best = min(gd_histories, key=lambda h: h[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(sa_best, label="SA (Best Run)")
    plt.plot(gd_best, label="GD (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Best Run Convergence - {name} (Best Params)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_best_params_convergence_exp{experiment_id}.png"))
    plt.close()

def run_experiments():
    # === Discrete Ising ===
    print("Running SA on discrete Ising (best params)")
    sa_disc_p = best_params["ising"]["ising_discrete"]["sa"]
    sa_disc = bootstrap_experiment(sa_discrete, num_runs, ising_energy, lattice_size=lattice_shape,
                                    T_init=sa_disc_p["T0"], alpha=sa_disc_p["alpha"], max_iter=20000,
                                    tol=tol, is_discrete=False, f=ising_energy)
    sa_disc_stats = sa_disc["stats"]
    if isinstance(sa_disc_stats, float):
        sa_disc_stats = {"near_optimal_proportion": sa_disc_stats}
    pd.DataFrame([{"Algorithm": "SA", **sa_disc_stats}]).to_csv(
        os.path.join(analytical_dir, f"summary_ising_discrete_best_exp{experiment_id}.csv"), index=False)

    # === Relaxed Ising ===
    print("Running SA and GD on relaxed Ising (best params)")
    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()
    init_point = np.random.uniform(-1, 1, size=lattice_shape).flatten()

    sa_p = best_params["ising"]["ising_relaxed"]["sa"]
    gd_p = best_params["ising"]["ising_relaxed"]["gd"]

    sa_relaxed = bootstrap_experiment(sa_continuous, num_runs, relaxed_f, x_init=init_point,
        T0=sa_p["T0"], alpha=sa_p["alpha"], step_size=sa_p["step_size"],
        max_iter=20000, tol=tol, bounds=[(-2, 2)] * (lattice_shape[0] * lattice_shape[1]),
        perturbation_method="normal", adaptive_step_size=False)

    gd_relaxed = bootstrap_experiment(gradient_descent, num_runs, relaxed_f, grad_wrapped,
        x_init=init_point, lr=gd_p["lr"], max_iter=20000, tol=tol)

    plot_combined_convergence("ising_relaxed", sa_relaxed["histories"], gd_relaxed["histories"])

    pd.DataFrame([
        {"Algorithm": "GD", **gd_relaxed["stats"]},
        {"Algorithm": "SA", **sa_relaxed["stats"]}
    ]).to_csv(os.path.join(analytical_dir, f"summary_ising_relaxed_best_exp{experiment_id}.csv"), index=False)

if __name__ == "__main__":
    run_experiments()
    print("All best-param Ising experiments complete.")
