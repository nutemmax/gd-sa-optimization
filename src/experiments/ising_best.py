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
from src.utils.utils_plots import plot_spin_evolution, plot_energy_trajectory, plot_final_spin_config
from src.utils.utils_experiments import bootstrap_experiment, get_experiment_id, evaluate_continuous_results

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
gridsearch_dir = os.path.join(results_dir, "gridsearch")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === Load best hyperparameters ===
with open(os.path.join(gridsearch_dir, "best_hyperparams.json")) as f:
    best_params = json.load(f)

# === Shared settings ===
num_runs = 50
tol = 1e-6
lattice_shape = (10, 10)
dim_ising = lattice_shape[0] * lattice_shape[1]
experiment_id = get_experiment_id()

def plot_combined_convergence(name, sa_histories, gd_histories, relaxed_f):
    sa_best = min(sa_histories, key=lambda h: relaxed_f(h[-1]))
    gd_best = min(gd_histories, key=lambda h: relaxed_f(h[-1]))

    sa_energies = [relaxed_f(x) for x in sa_best]
    gd_energies = [relaxed_f(x) for x in gd_best]

    plt.figure(figsize=(10, 6))
    plt.plot(sa_energies, label="SA (Best Run)", linewidth=2)
    plt.plot(gd_energies, label="GD (Best Run)", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Best Run Convergence - {name} (Best Params)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_best_convergence_exp{experiment_id}.png"))
    plt.close()

def run_experiments():
    # === Discrete Ising ===
    print("Running SA on discrete Ising (best params)")

    sa_disc_p = best_params["ising"]["ising_discrete"]["sa"]
    sa_disc = bootstrap_experiment(
        sa_discrete,
        num_runs,
        ising_energy,
        lattice_size=lattice_shape,
        T_init=sa_disc_p["T0"],
        alpha=sa_disc_p["alpha"],
        max_iter=20000,
        tol=tol,
        is_discrete=True,
        f=ising_energy
    )

    sa_disc["stats"] = evaluate_continuous_results(sa_disc["final_values"])
    best_sa_run = min(sa_disc['histories'], key=lambda h: ising_energy(h[-1]))

    print("Plotting final spin configuration...")
    plot_final_spin_config(best_sa_run[-1].reshape(lattice_shape), f"ising_discrete_best_exp{experiment_id}", plots_dir)

    print("Plotting energy trajectory...")
    plot_energy_trajectory(
        [ising_energy(x.reshape(lattice_shape)) for x in best_sa_run],
        f"ising_discrete_best_exp{experiment_id}",
        plots_dir
    )

    df_disc = pd.DataFrame([{"Algorithm": "SA", **sa_disc["stats"]}])
    df_disc.to_csv(
        os.path.join(analytical_dir, f"summary_ising_discrete_best_exp{experiment_id}.csv"),
        index=False
    )

    # === Relaxed Ising ===
    print("Running SA and GD on relaxed Ising (best params)")

    sa_p = best_params["ising"]["ising_relaxed"]["sa"]
    gd_p = best_params["ising"]["ising_relaxed"]["gd"]

    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()

    sa_histories = []
    gd_histories = []
    sa_final_values = []
    gd_final_values = []

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        x_init = np.random.uniform(-1, 1, size=dim_ising)

        sa_sol, sa_hist, sa_f_hist = sa_continuous(
            relaxed_f,
            x_init=x_init,
            T0=sa_p["T0"],
            alpha=sa_p["alpha"],
            step_size=sa_p["step_size"],
            max_iter=20000,
            tol=tol,
            bounds=[(-3, 3)] * dim_ising,
            perturbation_method="normal",
            adaptive_step_size=False
        )
        gd_sol, gd_hist, gd_f_hist = gradient_descent(
            relaxed_f,
            grad_wrapped,
            x_init=x_init,
            lr=gd_p["lr"],
            max_iter=20000,
            tol=tol
        )

        sa_histories.append(sa_hist)
        gd_histories.append(gd_hist)
        sa_final_values.append(sa_f_hist[-1])
        gd_final_values.append(gd_f_hist[-1])

    sa_stats = evaluate_continuous_results(sa_final_values)
    gd_stats = evaluate_continuous_results(gd_final_values)

    best_sa_run = sa_histories[np.argmin(sa_final_values)]
    best_gd_run = gd_histories[np.argmin(gd_final_values)]

    plot_energy_trajectory([relaxed_f(x) for x in best_sa_run],
                        f"ising_relaxed_SA_best_exp{experiment_id}", plots_dir)

    plot_combined_convergence("ising_relaxed", sa_histories, gd_histories, relaxed_f)

    df_relaxed = pd.DataFrame([
        {"Algorithm": "GD", **gd_stats},
        {"Algorithm": "SA", **sa_stats}
    ])
    df_relaxed.to_csv(
        os.path.join(analytical_dir, f"summary_ising_relaxed_best_exp{experiment_id}.csv"),
        index=False
    )

if __name__ == "__main__":
    run_experiments()
    print("All best-param Ising experiments complete.")
