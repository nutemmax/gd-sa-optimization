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
from src.utils.utils_experiments import bootstrap_experiment, get_experiment_id, evaluate_continuous_results, evaluate_discrete_results, generate_summary_csv

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === Shared settings ===
num_runs = 50
tol = 1e-6
lattice_shape = (10, 10)
experiment_id = get_experiment_id()
dim_ising = lattice_shape[0] * lattice_shape[1]

# === Baseline Hyperparameters ===
sa_disc_params = {"T_init": 50.0, "alpha": 0.99, "max_iter": 20000, "tol": tol}
sa_cont_params = {
    "T0": 50.0, "alpha": 0.99, "step_size": 0.1, "max_iter": 20000, "tol": tol,
    "bounds": [(-3, 3)] * dim_ising,
    "perturbation_method": "normal", "adaptive_step_size": False
}
gd_params = {"lr": 0.01, "max_iter": 20000}

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
        is_discrete=False,
        f=ising_energy
    )
    sa_disc["stats"] = evaluate_continuous_results(sa_disc["final_values"])

    best_sa_run = min(sa_disc['histories'], key=lambda h: ising_energy(h[-1]))
    plot_final_spin_config(best_sa_run[-1].reshape(lattice_shape), f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    print(f"Plotting final energy trajectory...")
    plot_energy_trajectory(
        [ising_energy(x.reshape(lattice_shape)) for x in best_sa_run],
        f"ising_discrete_baseline_exp{experiment_id}", plots_dir
    )
    df_disc = pd.DataFrame([{"Algorithm": "SA", **sa_disc["stats"]}])
    df_disc.to_csv(
        os.path.join(analytical_dir, f"summary_ising_discrete_baseline_exp{experiment_id}.csv"),
        index=False
    )

    # === Relaxed Ising: shared initialization ===
    print("Running SA and GD on relaxed Ising")

    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()

    sa_results = bootstrap_experiment(sa_continuous, num_runs, relaxed_f, x_init=None, f_star=-200.0, x_star=None, dim = dim_ising,**sa_cont_params)
    gd_x_init = np.random.uniform(-1, 1, size=lattice_shape[0] * lattice_shape[1])
    gd_results = bootstrap_experiment(
        gradient_descent,
        num_runs,
        relaxed_f,
        grad_wrapped,
        x_init=gd_x_init,
        f_star=-200.0,
        x_star=None,
        **gd_params)
    

    generate_summary_csv("ising_relaxed_baseline", gd_results["stats"], sa_results["stats"], experiment_id, analytical_dir)

    plot_combined_convergence("ising_relaxed", sa_results["histories"], gd_results["histories"], relaxed_f)

    # === Save final energies per run ===
    final_energy_data = []

    for i, val in enumerate(sa_results["final_values"]):
        final_energy_data.append({
            "Run": i + 1, "Algorithm": "SA", "FinalEnergy": val,
            "T0": sa_cont_params["T0"], "alpha": sa_cont_params["alpha"], "step_size": sa_cont_params["step_size"]
        })

    for i, val in enumerate(gd_results["final_values"]):
        final_energy_data.append({
            "Run": i + 1, "Algorithm": "GD", "FinalEnergy": val,
            "lr": gd_params["lr"]
        })

    df_final_energies = pd.DataFrame(final_energy_data)
    df_final_energies.to_csv(
        os.path.join(analytical_dir, f"final_energies_ising_relaxed_baseline_exp{experiment_id}.csv"),
        index=False
    )

    # === Plot histogram ===
    plt.figure(figsize=(10, 6))
    plt.hist(sa_results["final_values"], bins=20, alpha=0.6, label="SA")
    plt.hist(gd_results["final_values"], bins=20, alpha=0.6, label="GD")
    plt.xlabel("Final Energy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Final Energies (Baseline Params)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"hist_final_energies_ising_relaxed_baseline_exp{experiment_id}.png"))
    plt.close()


if __name__ == "__main__":
    run_experiments()
    print("All baseline Ising experiments complete.")
