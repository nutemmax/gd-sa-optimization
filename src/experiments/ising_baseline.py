import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from src.optimizers.sa import sa_continuous, sa_discrete
from src.optimizers.gd import gradient_descent
from src.problems.ising import ising_energy, relaxed_ising_energy, grad_relaxed_ising
from src.utils.utils_experiments import bootstrap_experiment, get_experiment_id, generate_summary_csv

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

# === Baseline Hyperparameters ===
sa_disc_params = {
    "T_init": 10.0,
    "alpha": 0.99,
    "max_iter": 10000,
    "tol": tol
}

sa_cont_params = {
    "T0": 10.0,
    "alpha": 0.99,
    "step_size": 0.1,
    "max_iter": 20000,
    "bounds": [(-2, 2)] * (lattice_shape[0] * lattice_shape[1]),
    "perturbation_method": "normal",
    "adaptive_step_size": False
}

gd_params = {
    "lr": 0.01,
    "max_iter": 20000
}

def plot_convergence(name, histories, label):
    best_idx = np.argmin([hist[-1] for hist in histories])
    plt.figure(figsize=(10, 6))
    plt.plot(histories[best_idx], label=f"{label} (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"{label} on {name} (Baseline)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_{label.lower()}_baseline_convergence_exp{experiment_id}.png"))
    plt.close()

def run_experiments():
    # === Ising Discrete: SA only ===
    print("Running SA on discrete Ising")
    sa_disc_results = bootstrap_experiment(
        sa_discrete,
        num_runs,
        ising_energy,
        state_init=None,
        lattice_size=lattice_shape,
        **sa_disc_params,
        is_discrete=True,
        best_state=None,
        f=ising_energy
    )
    generate_summary_csv("ising_discrete_baseline", gd_stats=None,
                         sa_stats={"near_optimal_proportion": sa_disc_results["stats"]},
                         experiment_id=experiment_id, save_dir=analytical_dir)
    plot_convergence("ising_discrete", sa_disc_results["histories"], "SA")

    # === Ising Relaxed: SA + GD ===
    print("Running GD and SA on relaxed Ising")

    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()
    ising_relaxed_init = np.random.uniform(-1, 1, size=lattice_shape).flatten()

    gd_results = bootstrap_experiment(
        gradient_descent,
        num_runs,
        relaxed_f,
        grad_wrapped,
        x_init=ising_relaxed_init,
        tol=tol,
        **gd_params
    )

    sa_results = bootstrap_experiment(
        sa_continuous,
        num_runs,
        relaxed_f,
        x_init=ising_relaxed_init,
        bounds=sa_cont_params["bounds"],
        T0=sa_cont_params["T0"],
        alpha=sa_cont_params["alpha"],
        step_size=sa_cont_params["step_size"],
        max_iter=sa_cont_params["max_iter"],
        perturbation_method=sa_cont_params["perturbation_method"],
        adaptive_step_size=sa_cont_params["adaptive_step_size"],
        tol=tol
    )

    generate_summary_csv("ising_relaxed_baseline",
                         gd_stats=gd_results['stats'],
                         sa_stats=sa_results['stats'],
                         experiment_id=experiment_id, save_dir=analytical_dir)

    plot_convergence("ising_relaxed", sa_results["histories"], "SA")
    plot_convergence("ising_relaxed", gd_results["histories"], "GD")

if __name__ == "__main__":
    run_experiments()
    print("All baseline Ising experiments complete.")
