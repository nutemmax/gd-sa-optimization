import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import matplotlib.pyplot as plt
from src.optimizers.sa import sa_continuous, sa_discrete
from src.optimizers.gd import gradient_descent
from src.problems.ising import ising_energy, relaxed_ising_energy, grad_relaxed_ising
from src.utils.utils_experiments import bootstrap_experiment, generate_summary_csv, get_experiment_id

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

def plot_convergence(name, histories, label):
    best_idx = np.argmin([hist[-1] for hist in histories])
    plt.figure(figsize=(10, 6))
    plt.plot(histories[best_idx], label=f"{label} (Best Run)")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"{label} on {name} (Best Params)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{name}_{label.lower()}_best_params_convergence_exp{experiment_id}.png"))
    plt.close()

def run_experiments():
    # === Ising Discrete: SA only ===
    print("Running SA on discrete Ising (best params)")
    sa_disc_p = best_params["ising"]["ising_discrete"]["sa"]

    sa_disc_results = bootstrap_experiment(
        sa_discrete,
        num_runs,
        ising_energy,
        state_init=None,
        lattice_size=lattice_shape,
        T_init=sa_disc_p["T0"],
        alpha=sa_disc_p["alpha"],
        max_iter=20000,
        tol=tol,
        is_discrete=True,
        best_state=None,
        f=ising_energy
    )
    generate_summary_csv("ising_discrete_best", gd_stats=None,
                         sa_stats={"near_optimal_proportion": sa_disc_results["stats"]},
                         experiment_id=experiment_id, save_dir=analytical_dir)
    plot_convergence("ising_discrete", sa_disc_results["histories"], "SA")

    # === Ising Relaxed: SA + GD ===
    print("Running GD and SA on relaxed Ising (best params)")
    relaxed_f = lambda x: relaxed_ising_energy(x.reshape(lattice_shape))
    grad_wrapped = lambda x: grad_relaxed_ising(x.reshape(lattice_shape)).flatten()
    ising_relaxed_init = np.random.uniform(-1, 1, size=lattice_shape).flatten()

    sa_cont_p = best_params["ising"]["ising_relaxed"]["sa"]
    gd_p = best_params["ising"]["ising_relaxed"]["gd"]

    gd_results = bootstrap_experiment(
        gradient_descent,
        num_runs,
        relaxed_f,
        grad_wrapped,
        x_init=ising_relaxed_init,
        lr=gd_p["lr"],
        max_iter=20000,
        tol=tol
    )

    sa_results = bootstrap_experiment(
        sa_continuous,
        num_runs,
        relaxed_f,
        x_init=ising_relaxed_init,
        T0=sa_cont_p["T0"],
        alpha=sa_cont_p["alpha"],
        step_size=sa_cont_p["step_size"],
        max_iter=20000,
        tol=tol,
        bounds=[(-2, 2)] * (lattice_shape[0] * lattice_shape[1]),
        perturbation_method="normal",
        adaptive_step_size=False
    )

    generate_summary_csv("ising_relaxed_best",
                         gd_stats=gd_results['stats'],
                         sa_stats=sa_results['stats'],
                         experiment_id=experiment_id, save_dir=analytical_dir)

    plot_convergence("ising_relaxed", sa_results["histories"], "SA")
    plot_convergence("ising_relaxed", gd_results["histories"], "GD")

if __name__ == "__main__":
    run_experiments()
    print("All best-param Ising experiments complete.")
