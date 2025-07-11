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
from src.utils.utils_experiments import bootstrap_experiment_ising, get_experiment_id, evaluate_ising_results, generate_summary_csv

# === paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(base_dir, "results")
plots_dir = os.path.join(results_dir, "plots")
analytical_dir = os.path.join(results_dir, "analytical")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(analytical_dir, exist_ok=True)

# === shared settings ===
num_runs = 50
tol = 1e-6
lattice_shape = (10, 10)
experiment_id = get_experiment_id()
dim_ising = lattice_shape[0]

# === baseline hyperparameters ===
sa_disc_params = {"T_init": 50.0, "alpha": 0.99, "max_iter": 20000, "tol": tol}
sa_cont_params = {
    "T0": 50.0, "alpha": 0.99, "step_size": 0.1, "max_iter": 20000, "tol": tol,
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
    # === discrete ising ===
    print("Running SA on discrete Ising")
    x_inits_disc = [np.random.choice([-1, 1], size=lattice_shape) for _ in range(num_runs)]

    sa_disc = bootstrap_experiment_ising(
        algorithm_function=sa_discrete,
        runs=num_runs,
        f=ising_energy,
        dim=dim_ising,
        x_inits=x_inits_disc,
        is_discrete=True,
        best_state=None,
        hamming_threshold=0,
        f_star=-200.0,
        **sa_disc_params
    )

    best_sa_run = min(sa_disc['histories'], key=lambda h: ising_energy(h[-1]))
    plot_final_spin_config(best_sa_run[-1].reshape(lattice_shape), f"Ising Discrete Baseline", f"ising_discrete_baseline_exp{experiment_id}", plots_dir)
    plot_energy_trajectory(
        [ising_energy(x.reshape(lattice_shape)) for x in best_sa_run],
        f"ising_discrete_baseline_exp{experiment_id}", plots_dir
    )
    df_disc = pd.DataFrame([{"Algorithm": "SA", **sa_disc["stats"]}])
    df_disc.to_csv(
        os.path.join(analytical_dir, f"summary_ising_discrete_baseline_exp{experiment_id}.csv"),
        index=False
    )

    # === relaxed ising ===
    print("Running SA and GD on relaxed Ising")
    x_inits_cont = [np.random.uniform(-1, 1, size=lattice_shape) for _ in range(num_runs)]
    x_inits_flat = [x.flatten() for x in x_inits_cont]

    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()

    sa_results = bootstrap_experiment_ising(
        algorithm_function=sa_continuous,
        runs=num_runs,
        f=relaxed_f,
        dim=dim_ising,
        x_inits=x_inits_cont,
        f_star=-200.0,
        x_star=None,
        init_range=(-1, 1),
        **sa_cont_params
    )

    gd_results = bootstrap_experiment_ising(
        algorithm_function=gradient_descent,
        runs=num_runs,
        f=relaxed_f,
        grad_f=grad_wrapped,
        dim=dim_ising,
        x_inits=x_inits_flat,
        f_star=-200.0,
        x_star=None,
        **gd_params
    )

    generate_summary_csv("ising_relaxed_baseline", gd_results["stats"], sa_results["stats"], experiment_id, analytical_dir)
    plot_combined_convergence("ising_relaxed", sa_results["histories"], gd_results["histories"], relaxed_f)

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
