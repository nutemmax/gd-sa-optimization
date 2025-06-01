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
    Computes:
    - mean, best, worst, std
    - epsilon based on best value
    - near-optimal count
    - MAE, MSE, RMSE (against f_star)
    - mean_runtime (if runtimes provided)
    - mean_distance_to_x_star (if final_states and x_star provided)
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

    # Mean distance to x*
    if final_states is not None and x_star is not None:
        distances = [np.linalg.norm(x - x_star) for x in final_states]
        mean_dist_x_star = np.mean(distances)
    else:
        mean_dist_x_star = None

    return {
        'rmse': rmse,
        'best': best_value,
        'worst': np.max(final_values),
        'std': np.std(final_values),
        'epsilon': epsilon,
        'near_optimal_count': int(near_optimal_count),
        'mean': np.mean(final_values),
        'mse': mse,
        'mae': mae,
        'mean_runtime_sec': mean_runtime,
        'mean_dist_to_x_star': mean_dist_x_star
    }


def evaluate_discrete_results(final_values, final_states, best_state, threshold=0, runtimes=None, f_star=-200.0):

    final_values = np.array(final_values)

    # basic statistics
    mean_value = np.mean(final_values)
    best_value = np.min(final_values)
    worst_value = np.max(final_values)
    std_value = np.std(final_values)

    # error metrics
    mse = np.mean((final_values - f_star) ** 2)
    rmse = np.sqrt(mse)

    # hamming distances
    distances = [np.sum(state != best_state) for state in final_states]
    mean_hamming = np.mean(distances)

    # Runtime
    mean_runtime = np.mean(runtimes) if runtimes is not None else None

    return {
        "mean": mean_value,
        "best": best_value,
        "worst": worst_value,
        "std": std_value,
        "mse": mse,
        "rmse": rmse,
        "mean_hamming_dist": mean_hamming,
        "mean_runtime": mean_runtime
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
        if grad_f is not None:
            result = algorithm_function(f, grad_f, x_init=x_init, init_range=init_range, **kwargs)
        else:
            result = algorithm_function(f, x_init=x_init, init_range=init_range, **kwargs)

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
    grad_f = grad_relaxed_ising,
    dim = 10,
    x_inits = None, 
    is_discrete=False,
    best_state=None,
    hamming_threshold=0,
    f_star=0.0,
    x_star=None,
    name = "ising",
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

        # initialization
        x_init = x_inits[i - 1] if x_inits is not None else (
            np.random.choice([-1, 1], size=(dim, dim)) if is_discrete else np.random.uniform(-1, 1, size=(dim, dim))
        )
        kwargs.pop("x_init", None)  # Remove just in case

        if is_discrete:
            result = algorithm_function(f, x_init=x_init, **kwargs)
        else:
            if algorithm_function.__name__ == "gradient_descent":
                result = algorithm_function(f, grad_f, x_init=x_init, **kwargs)
            else:  # for sa_continuous or anything that does NOT take grad_f
                result = algorithm_function(f, x_init=x_init, **kwargs)

        x_final, x_hist, f_hist = result

        duration = time.time() - start_time
        final_values.append(f_hist[-1])
        final_states.append(x_final)
        state_histories.append(x_hist)
        energy_histories.append(f_hist)
        runtimes.append(duration)

    if is_discrete:
        stats = evaluate_discrete_results(
            final_values=final_values,
            final_states=final_states,
            best_state=best_state,
            threshold=hamming_threshold,
            runtimes=runtimes,
            f_star=f_star
        )
        epsilon = None
    else:
        stats = evaluate_continuous_results(
            final_values=final_values,
            runtimes=runtimes,
            final_states=final_states,
            x_star=x_star,
            f_star=f_star
        )
        epsilon = stats['epsilon']


    return {
        'final_values': final_values,
        'final_states': final_states,
        'stats': stats,
        'epsilon': epsilon,
        'histories': state_histories,
        'f_histories': energy_histories,
        'runtimes': runtimes
    }


def bootstrap_experiment_old(
    algorithm_function,
    runs,
    *args,
    f=None,
    is_discrete=False,
    best_state=None,
    hamming_threshold=0,
    x_init_strategy=None,
    dim=None,
    f_star=0.0,
    x_star=None,
    **kwargs
):
    """
    Runs the optimization algorithm multiple times and collects summary statistics.

    Returns:
        dict: {
            'final_values': [...],
            'final_states': [...],
            'stats': {...},
            'epsilon': float or None,
            'near_optimal_count': int or None,
            'state_histories': [...],
            'energy_histories': [...],
            'runtimes': [...],
        }
    """
    final_values = []
    final_states = []
    state_histories = []
    energy_histories = []
    runtimes = []

    # fetch init_range from kwargs only once
    # init_range = kwargs.get("init_range", (-5,5))
    
    if f is not None and hasattr(f, '__name__'):
        fname = f.__name__.lower()
        print(f"function name : {fname}")
        if fname == "rosenbrock":
            init_range = (-2, 2)
        elif fname.startswith("ising"):
            init_range = (-1, 1)
        else:
            init_range = (-5, 5)
    else:
        init_range = (-5, 5)

    for i in range(1, runs + 1):
        print(f'Run {i}/{runs}...', flush=True)
        start_time = time.time()

        current_kwargs = kwargs.copy()

        # custom init strategy
        if "x_init" in current_kwargs and current_kwargs["x_init"] is None:
            inferred_dim = dim if dim is not None else 2
            current_kwargs["x_init"] = np.random.uniform(*init_range, size=inferred_dim)
        
        final_solution, history, f_history = algorithm_function(*args, **current_kwargs)

        duration = time.time() - start_time
        final_value = f_history[-1]

        final_values.append(final_value)
        final_states.append(final_solution)
        state_histories.append(history)
        energy_histories.append(f_history)
        runtimes.append(duration)

    if is_discrete:
        stats = evaluate_discrete_results(
            final_states=final_states,
            best_state=best_state,
            threshold=hamming_threshold,
            runtimes=runtimes
        )
        epsilon = None
        near_optimal_count = None
    else:
        stats = evaluate_continuous_results(
            final_values=final_values,
            runtimes=runtimes,
            final_states=final_states,
            x_star=x_star,
            f_star=f_star
        )
        epsilon = stats['epsilon']
        near_optimal_count = stats['near_optimal_count']

    return {
        'final_values': final_values,
        'final_states': final_states,
        'stats': stats,
        'epsilon': epsilon,
        'near_optimal_count': near_optimal_count,
        'histories': state_histories,
        'f_histories': energy_histories,
        'runtimes': runtimes
    }


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

def generate_summary_csv(benchmark_name, gd_stats, sa_stats, experiment_id, save_dir):
    """
    Saves one-line-per-algorithm summary statistics for a benchmark.
    """
    summary_data = []
    if gd_stats is not None:
        summary_data.append({"Algorithm": "GD", **gd_stats})
    if sa_stats is not None:
        summary_data.append({"Algorithm": "SA", **sa_stats})

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
