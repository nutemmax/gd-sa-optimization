import numpy as np
import os
import time
import pandas as pd
import time


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.problems.ising import ising_energy, relaxed_ising_energy, grad_relaxed_ising

# -------------------------
# Core Evaluation Functions
# -------------------------

def evaluate_continuous_results(final_values, runtimes=None, final_states=None, x_star=None, f_star=0.0):
    """
    Evaluates results from continuous optimization runs.

    Parameters:
    - final_values (list or array): Final function values from each run.
    - runtimes (list, optional): Runtime durations for each run, used to compute average runtime.
    - final_states (list or array, optional): Final states (solution vectors) from each run.
    - x_star (array-like, optional): Known global minimizer, used to compute distance metrics.
    - f_star (float, default=0.0): Known global minimum value, used for error metrics.

    Returns:
    dict with:
    - 'mean': Mean final function value
    - 'best': Best (lowest) function value
    - 'worst': Worst (highest) function value
    - 'std': Standard deviation of function values
    - 'epsilon': Precision threshold around best value (adaptive)
    - 'near_optimal_count': Number of runs within epsilon of best value
    - 'mae': Mean Absolute Error relative to f_star
    - 'mse': Mean Squared Error relative to f_star
    - 'rmse': Root Mean Squared Error relative to f_star
    - 'mean_runtime_sec': Average runtime per run (if runtimes provided)
    - 'med': Mean Euclidean Distance to x_star (if final_states and x_star provided)
    - 'rmed': Root Mean Euclidean Distance to x_star (if final_states and x_star provided)
    """
    if not final_values:
        return {}

    final_values = np.array(final_values)
    best_value = np.min(final_values)

    # Epsilon logic
    if best_value > 1:
        epsilon = 0.01
    else:
        s = f"{best_value:.10f}"
        decimals = s.split('.')[1]
        k = next((i for i, ch in enumerate(decimals) if ch != '0'), len(decimals))
        epsilon = 0.01 if k == 0 else 10 ** (-(k + 1))

    near_optimal_count = np.sum(np.abs(final_values - best_value) <= epsilon)

    # Error metrics (vs f_star)
    mse = np.mean((final_values - f_star) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(final_values - f_star))

    # Runtime
    mean_runtime = np.mean(runtimes) if runtimes else None

    # mean euclidean distance to x*
    if final_states is not None and x_star is not None:
        distances = [np.linalg.norm(x - x_star)**2 for x in final_states]
        med = np.mean(distances)
        rmed = np.sqrt(np.mean([d**2 for d in distances]))
    else:
        med = None
        rmed = None

    return {
        'rmse': rmse,
        'rmed' : rmed,
        'best': best_value,
        'worst': np.max(final_values),
        'std': np.std(final_values),
        'epsilon': epsilon,
        'near_optimal_count': int(near_optimal_count),
        'mean': np.mean(final_values),
        'mse': mse,
        'med' : med,
        'mae': mae,
        'mean_runtime_sec': mean_runtime,
    }


def evaluate_ising_results(final_values, final_states, best_state=None, threshold=0, runtimes=None, f_star=-200.0):
    """
    Evaluates Ising model optimization results (discrete or relaxed).

    Computes:
    - mean, best, worst, std
    - MSE, RMSE relative to f_star
    - mean runtime (if runtimes provided)
    - mean Hamming distance (if best_state provided)
    - MED and RMED = mean and root mean squared distance to closest of (+1,...,+1) or (-1,...,-1)
    """
    final_values = np.array(final_values)

    # Basic stats
    mean_value = np.mean(final_values)
    best_value = np.min(final_values)
    worst_value = np.max(final_values)
    std_value = np.std(final_values)

    # Errors vs known energy minimum
    mse = np.mean((final_values - f_star) ** 2)
    rmse = np.sqrt(mse)
    mean_runtime = np.mean(runtimes) if runtimes is not None else None

    # Mean Hamming distance (only for discrete Ising)
    if best_state is not None:
        hamming_distances = [np.sum(state != best_state) for state in final_states]
        mean_hamming = np.mean(hamming_distances)
    else:
        mean_hamming = None

    # Distance to closest global optimum: (+1,...,+1) or (-1,...,-1)
    ref_plus = np.ones_like(final_states[0])
    ref_minus = -np.ones_like(final_states[0])
    dists = [min(np.linalg.norm(state - ref_plus), np.linalg.norm(state - ref_minus)) for state in final_states]
    med = np.mean(dists)
    rmed = np.sqrt(np.mean(np.square(dists)))

    return {
        "rmse": rmse,
        "rmed": rmed,
        "best": best_value,
        "worst": worst_value,
        "std": std_value,
        "mean": mean_value,
        "mse": mse,
        "mean_runtime": mean_runtime,
        "mean_hamming_dist": mean_hamming,
        "med": med,
        
    }


def compute_frequency_continuous(final_states, best_state, f):
    """
    Count how many states are within epsilon of the best objective value.
    Epsilon is computed based on leading decimals of best value.
    """
    best_value = f(best_state)
    if best_value > 1:
        epsilon = 0.01
    else:
        s = f"{best_value:.10f}"
        decimals = s.split('.')[1]
        k = next((i for i, ch in enumerate(decimals) if ch != '0'), len(decimals))
        epsilon = 0.01 if k == 0 else 10 ** (-(k + 1))

    count = sum(1 for state in final_states if abs(f(state) - best_value) <= epsilon)
    return count, epsilon

# --------------------------------
# Bootstrapping
# --------------------------------
def bootstrap_experiment_benchmarks(
    algorithm_function,
    runs,
    f,
    grad_f=None,
    dim=2,
    x_inits = None,
    f_star=0.0,
    x_star=None,
    name="benchmark",
    **kwargs
):
    final_values = []
    final_states = []
    state_histories = []
    energy_histories = []
    runtimes = []

    # Initialization range
    if name.lower() == "rosenbrock":
        init_range = (-2, 2)
    else:
        init_range = (-5, 5)

    for i in range(1, runs + 1):
        print(f"[Benchmark] Run {i}/{runs}", flush=True)
        start_time = time.time()

        # pre-specified init if provided
        x_init = x_inits[i - 1] if x_inits is not None else np.random.uniform(*init_range, size=dim)

        # call the algorithm, passing init_range for GD to enable clipping
        if algorithm_function.__name__ == "sa_gd_hybrid":
            result = algorithm_function(f=f, grad_f=grad_f, x_init=x_init, name=name, **kwargs)
        elif grad_f is not None:
            result = algorithm_function(f, grad_f, x_init=x_init, init_range=init_range, name = name, **kwargs)
        else:
            result = algorithm_function(f, x_init=x_init, init_range=init_range, name = name, **kwargs)


        x_final, x_hist, f_hist = result
        duration = time.time() - start_time

        final_values.append(f_hist[-1])
        final_states.append(x_final)
        state_histories.append(x_hist)
        energy_histories.append(f_hist)
        runtimes.append(duration)

    stats = evaluate_continuous_results(
        final_values=final_values,
        runtimes=runtimes,
        final_states=final_states,
        x_star=x_star,
        f_star=f_star
    )

    return {
        'final_values': final_values,
        'final_states': final_states,
        'stats': stats,
        'epsilon': stats['epsilon'],
        'near_optimal_count': stats['near_optimal_count'],
        'histories': state_histories,
        'f_histories': energy_histories,
        'runtimes': runtimes
    }


def bootstrap_experiment_ising(
    algorithm_function,
    runs,
    f,
    grad_f=grad_relaxed_ising,
    dim=10,
    x_inits=None,
    is_discrete=False,
    best_state=None,
    hamming_threshold=0,
    f_star=0.0,
    x_star=None,
    name="ising",
    **kwargs
):
    final_values = []
    final_states = []
    state_histories = []
    energy_histories = []
    runtimes = []

    for i in range(1, runs + 1):
        print(f"[Ising {'Discrete' if is_discrete else 'Relaxed'}] Run {i}/{runs}", flush=True)
        start_time = time.time()

        # Select initialization
        x_init = x_inits[i - 1] if x_inits is not None else (
            np.random.choice([-1, 1], size=(dim, dim)) if is_discrete else np.random.uniform(-1, 1, size=(dim, dim))
        )
        kwargs.pop("x_init", None)  # prevent accidental override
        kwargs_clean = kwargs.copy()

        # Execute algorithm
        if is_discrete:
            result = algorithm_function(f, x_init=x_init, **kwargs_clean)
        else:
            if algorithm_function.__name__ in ["gradient_descent", "sa_gd_hybrid"]:
                result = algorithm_function(f=f, grad_f=grad_f, x_init=x_init, name=name, **kwargs_clean)
            else:
                result = algorithm_function(f, x_init=x_init, **kwargs_clean)


        x_final, x_hist, f_hist = result

        # Record metrics
        duration = time.time() - start_time
        final_values.append(f_hist[-1])
        final_states.append(x_final)
        state_histories.append(x_hist)
        energy_histories.append(f_hist)
        runtimes.append(duration)

    # Use a unified evaluation function for both cases
    stats = evaluate_ising_results(
        final_values=final_values,
        final_states=final_states,
        f_star=f_star,
        best_state=best_state,
        threshold=hamming_threshold,
        runtimes=runtimes
    )

    return {
        "final_values": final_values,
        "final_states": final_states,
        "stats": stats,
        "epsilon": None,  # not applicable for Ising experiments
        "histories": state_histories,
        "f_histories": energy_histories,
        "runtimes": runtimes
    }


# def bootstrap_experiment_ising(
#     algorithm_function,
#     runs,
#     f,
#     grad_f = grad_relaxed_ising,
#     dim = 10,
#     x_inits = None, 
#     is_discrete=False,
#     best_state=None,
#     hamming_threshold=0,
#     f_star=0.0,
#     x_star=None,
#     name = "ising",
#     **kwargs
# ):
#     final_values = []
#     final_states = []
#     state_histories = []
#     energy_histories = []
#     runtimes = []

#     for i in range(1, runs + 1):
#         print(f"[Ising {'Discrete' if is_discrete else 'Relaxed'}] Run {i}/{runs}", flush=True)
#         start_time = time.time()

#         # initialization
#         x_init = x_inits[i - 1] if x_inits is not None else (
#             np.random.choice([-1, 1], size=(dim, dim)) if is_discrete else np.random.uniform(-1, 1, size=(dim, dim))
#         )
#         kwargs.pop("x_init", None)  # Remove just in case

#         if is_discrete:
#             result = algorithm_function(f, x_init=x_init, **kwargs)
#         else:
#             if algorithm_function.__name__ == "gradient_descent":
#                 result = algorithm_function(f, grad_f, x_init=x_init, **kwargs)
#             else:  # for sa_continuous or anything that does NOT take grad_f
#                 result = algorithm_function(f, x_init=x_init, **kwargs)

#         x_final, x_hist, f_hist = result

#         duration = time.time() - start_time
#         final_values.append(f_hist[-1])
#         final_states.append(x_final)
#         state_histories.append(x_hist)
#         energy_histories.append(f_hist)
#         runtimes.append(duration)

#     if is_discrete:
#         stats = evaluate_ising_results(
#             final_values=final_values,
#             final_states=final_states,
#             best_state=best_state,
#             threshold=hamming_threshold,
#             runtimes=runtimes,
#             f_star=f_star
#         )
#         epsilon = None
#     else:
#         stats = evaluate_continuous_results(
#             final_values=final_values,
#             runtimes=runtimes,
#             final_states=final_states,
#             x_star=x_star,
#             f_star=f_star
#         )
#         epsilon = stats['epsilon']


#     return {
#         'final_values': final_values,
#         'final_states': final_states,
#         'stats': stats,
#         'epsilon': epsilon,
#         'histories': state_histories,
#         'f_histories': energy_histories,
#         'runtimes': runtimes
#     }



# --------------------------------
# Experiment ID & Result Saving
# --------------------------------

def get_experiment_id() -> int :
    """
    Reads the current experiment ID from file, increments it, and returns the new ID.
    """
    path = "experiment_id.txt"
    if os.path.exists(path):
        with open(path, 'r') as f:
            current_id = int(f.read().strip())
    else:
        current_id = 0
    new_id = current_id + 1
    with open(path, 'w') as f:
        f.write(str(new_id))
    return new_id

def generate_summary_csv(benchmark_name, gd_stats, sa_stats, hybrid_stats, experiment_id, save_dir, name_alg = None):
    """
    Saves one-line-per-algorithm summary statistics for a benchmark.
    """
    summary_data = []
    if gd_stats is not None:
        summary_data.append({"Algorithm": "GD", **gd_stats})
    if sa_stats is not None:
        summary_data.append({"Algorithm": "SA", **sa_stats})
    if hybrid_stats is not None:
        summary_data.append({"Algorithm": f"{name_alg}", **hybrid_stats})

    summary_df = pd.DataFrame(summary_data)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"summary_{benchmark_name}_exp{experiment_id}.csv")
    summary_df.to_csv(path, index=False)
    print(f"✅ Summary CSV saved to: {path}")


def save_convergence_histories(histories, filename_prefix, save_dir):
    """
    Saves each convergence curve (f_history) from bootstrapping to a CSV file.
    
    Parameters:
        histories (list of list): Each inner list contains function values per iteration.
        filename_prefix (str): e.g., 'GD_rastrigin_exp3'
        save_dir (str): Directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, hist in enumerate(histories):
        path = os.path.join(save_dir, f"{filename_prefix}_run{idx}.csv")
        np.savetxt(path, hist, delimiter=",")



# --------------------------------
# OTHER
# --------------------------------
def set_seed(seed=42):
    np.random.seed(seed)

def hamming_distance(state1, state2):
    """
    Computes the Hamming distance between two discrete states.
    Parameters:
        state1, state2 (np.array): Discrete states (e.g., 2D arrays with values -1 or 1).
    Returns:
        int: The number of positions where state1 and state2 differ.
    """
    return np.sum(state1 != state2)

def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))
