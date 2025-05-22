# src/experiments/ising_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.optimizers.sa import sa_continuous, sa_discrete
from src.optimizers.gd import gradient_descent
from src.problems.ising import ising_energy, relaxed_ising_energy, grad_relaxed_ising
from src.utils.utils_plots import plot_spin_evolution, plot_energy_trajectory, plot_final_spin_config
from src.utils.utils_experiments import bootstrap_experiment, get_experiment_id, evaluate_discrete_results



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

# === Baseline Hyperparameters ===
sa_disc_params = {"T_init": 10.0, "alpha": 0.99, "max_iter": 10000, "tol": tol}
sa_cont_params = {
    "T0": 10.0, "alpha": 0.99, "step_size": 0.1, "max_iter": 20000, "tol": tol,
    "bounds": [(-2, 2)] * (lattice_shape[0] * lattice_shape[1]),
    "perturbation_method": "normal", "adaptive_step_size": False
}
gd_params = {"lr": 0.01, "max_iter": 20000}

def plot_combined_convergence(name, sa_histories, gd_histories, relaxed_f):
    # get SA and GD run with lowest final energy
    sa_best = min(sa_histories, key=lambda h: relaxed_f(h[-1]))
    gd_best = min(gd_histories, key=lambda h: relaxed_f(h[-1]))

    # compute energy per iteration for those two runs only
    sa_energies = [relaxed_f(x) for x in sa_best]
    gd_energies = [relaxed_f(x) for x in gd_best]

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(sa_energies, label="SA (Best Run)", linewidth=2)
    plt.plot(gd_energies, label="GD (Best Run)", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Best Run Convergence - {name} (Baseline)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_baseline_convergence_exp{experiment_id}.png"))
    plt.close()


def run_experiments():
    # === Discrete Ising ===
    print("Running SA on discrete Ising")
    sa_disc = bootstrap_experiment(
        sa_discrete,
        num_runs,
        ising_energy,
        lattice_size=lattice_shape,
        **sa_disc_params,
        is_discrete=True,
        f=ising_energy,
        best_state=None  # Temporary
    )

    # Compute best run and update stats now that best_state is known
    best_sa_run = min(sa_disc['histories'], key=lambda h: ising_energy(h[-1]))
    sa_disc["stats"] = evaluate_discrete_results(sa_disc["final_states"], best_sa_run[-1])

    print(f"Plotting final spin configuration....")
    plot_final_spin_config(best_sa_run[-1].reshape(lattice_shape), f"ising_discrete_baseline_exp{experiment_id}", plots_dir)

    print(f"Plotting energy trajectory....")
    plot_energy_trajectory([ising_energy(config.reshape(lattice_shape)) for config in best_sa_run],
                           f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    if isinstance(sa_disc["stats"], float):
        df = pd.DataFrame([{"Algorithm": "SA", "near_optimal_proportion": sa_disc["stats"]}])
    else:
        df = pd.DataFrame([{"Algorithm": "SA", **sa_disc["stats"]}])
    df.to_csv(os.path.join(analytical_dir, f"summary_ising_discrete_baseline_exp{experiment_id}.csv"), index=False)

    # === Relaxed Ising: GD and SA ===
    print("Running SA and GD on relaxed Ising")
    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()

    # Define initialization strategy
    dim_ising = lattice_shape[0] * lattice_shape[1]
    init_strategy = lambda d: np.random.uniform(-1, 1, size=d)


    sa_relaxed = bootstrap_experiment(
        sa_continuous,
        num_runs,
        relaxed_f,
        x_init=None,
        x_init_strategy=init_strategy,
        dim=dim_ising,
        **sa_cont_params
    )

    gd_relaxed = bootstrap_experiment(
        gradient_descent,
        num_runs,
        relaxed_f,
        grad_wrapped,
        x_init=None,
        x_init_strategy=init_strategy,
        dim=dim_ising,
        **gd_params,
        tol=tol
    )


    # Plot best runs
    best_sa_relaxed = min(sa_relaxed["histories"], key=lambda h: relaxed_f(h[-1]))
    best_gd_relaxed = min(gd_relaxed["histories"], key=lambda h: relaxed_f(h[-1]))

    plot_energy_trajectory([relaxed_f(x) for x in best_sa_relaxed],
                           f"ising_relaxed_SA_baseline_exp{experiment_id}", plots_dir)

    plot_combined_convergence("ising_relaxed", sa_relaxed["histories"], gd_relaxed["histories"], relaxed_f)

    df_relaxed = pd.DataFrame([
        {"Algorithm": "GD", **gd_relaxed["stats"]},
        {"Algorithm": "SA", **sa_relaxed["stats"]}
    ])
    df_relaxed.to_csv(os.path.join(analytical_dir, f"summary_ising_relaxed_baseline_exp{experiment_id}.csv"), index=False)


def run_experiments_olddd():
    # === Discrete Ising ===
    print("Running SA on discrete Ising")
    sa_disc = bootstrap_experiment(
        sa_discrete,
        num_runs,
        ising_energy,
        lattice_size=lattice_shape,
        **sa_disc_params,
        is_discrete=True,
        f=ising_energy,
        best_state=best_sa_run[-1]
    )

    best_sa_run = min(sa_disc['histories'], key=lambda h: ising_energy(h[-1]))
    # print(f"Plotting spin evolution....")
    # plot_spin_evolution([state.reshape(lattice_shape) for state in best_sa_run], lattice_shape, f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    print(f"Plotting final spin configuration....")
    plot_final_spin_config(best_sa_run[-1].reshape(lattice_shape), f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    print(f"Plotting energy trajectory....")
    plot_energy_trajectory([ising_energy(config.reshape(lattice_shape)) for config in best_sa_run],
                        f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    sa_disc_stats = sa_disc['stats']

    if isinstance(sa_disc_stats, float):
        sa_disc_stats = {"near_optimal_proportion": sa_disc_stats}

    df = pd.DataFrame([{"Algorithm": "SA", **sa_disc_stats}])
    df.to_csv(os.path.join(analytical_dir, f"summary_ising_discrete_baseline_exp{experiment_id}.csv"), index=False)

    # === Relaxed Ising: GD and SA ===
    print("Running SA and GD on relaxed Ising")
    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()
    sa_relaxed = bootstrap_experiment(sa_continuous, num_runs, relaxed_f, x_init=None, **sa_cont_params)
    gd_relaxed = bootstrap_experiment(gradient_descent, num_runs, relaxed_f, grad_wrapped,
                                       x_init=None, **gd_params, tol=tol)
    
    # === Additional Plots for Relaxed Ising ===
    best_sa_relaxed = min(sa_relaxed["histories"], key=lambda h: relaxed_f(h[-1]))
    best_gd_relaxed = min(gd_relaxed["histories"], key=lambda h: relaxed_f(h[-1]))

    plot_energy_trajectory([relaxed_f(x) for x in best_sa_relaxed],
                        f"ising_relaxed_SA_baseline_exp{experiment_id}", plots_dir)
    plot_combined_convergence("ising_relaxed", sa_relaxed["histories"], gd_relaxed["histories"], relaxed_f)


    df_relaxed = pd.DataFrame([
        {"Algorithm": "GD", **gd_relaxed["stats"]},
        {"Algorithm": "SA", **sa_relaxed["stats"]}
    ])
    df_relaxed.to_csv(os.path.join(analytical_dir, f"summary_ising_relaxed_baseline_exp{experiment_id}.csv"), index=False)

if __name__ == "__main__":
    run_experiments()
    print("All baseline Ising experiments complete.")
