import numpy as np
import os
import time
import pandas as pd
import time

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


def evaluate_discrete_results(final_states, best_state, threshold=0, runtimes=None):
    """
    Evaluates results from discrete optimization runs using Hamming distance to best_state.

    Computes:
    - mean_hamming_distance
    - count of states within Hamming distance threshold
    - mean_runtime_sec (if runtimes provided)

    Args:
        final_states (list of np.array): list of binary spin configurations
        best_state (np.array): known best configuration (ground truth)
        threshold (int): Hamming distance threshold for "near-optimal"
        runtimes (list of float): list of runtimes (optional)

    Returns:
        dict with keys:
            - mean_hamming_dist
            - near_optimal_count
            - mean_runtime_sec (if runtimes provided)
    """
    if best_state is None or not final_states:
        return {}

    distances = [np.sum(state != best_state) for state in final_states]
    mean_hamming = np.mean(distances)
    near_optimal_count = sum(d <= threshold for d in distances)

    mean_runtime = np.mean(runtimes) if runtimes else None

    return {
        'mean_hamming_dist': mean_hamming,
        'near_optimal_count': near_optimal_count,
        'mean_runtime_sec': mean_runtime
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

def bootstrap_experiment(
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
    init_range = kwargs.pop("init_range", (-5, 5))

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
