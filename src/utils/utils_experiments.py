import numpy as np
import os
import time
import pandas as pd

# -------------------------
# Core Evaluation Functions
# -------------------------

def evaluate_continuous_results(final_values):
    """
    Evaluates results from multiple continuous optimization runs.

    Computes:
    - mean
    - best (min)
    - worst (max)
    - std
    - epsilon (based on leading decimals of best value)
    - count of results within epsilon of best

    Returns:
        dict: with keys 'mean', 'best', 'worst', 'std', 'epsilon', 'near_optimal_count'
    """
    if not final_values:
        return {}

    final_values = np.array(final_values)
    best_value = np.min(final_values)

    # Compute epsilon based on best value
    if best_value > 1:
        epsilon = 0.01
    else:
        s = f"{best_value:.10f}"
        decimals = s.split('.')[1]
        k = next((i for i, ch in enumerate(decimals) if ch != '0'), len(decimals))
        epsilon = 0.01 if k == 0 else 10 ** (-(k + 1))

    near_optimal_count = np.sum(np.abs(final_values - best_value) <= epsilon)

    return {
        'mean': np.mean(final_values),
        'best': best_value,
        'worst': np.max(final_values),
        'std': np.std(final_values),
        'epsilon': epsilon,
        'near_optimal_count': int(near_optimal_count)
    }

def evaluate_discrete_results(final_states, best_state, threshold=0):
    """
    Compute proportion of final states within Hamming distance `threshold` of best_state.
    """
    count = sum(1 for state in final_states if np.sum(state != best_state) <= threshold)
    return count / len(final_states) if final_states else 0.0

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
    **kwargs
):
    """
    Runs the optimization algorithm multiple times and collects summary statistics.

    Parameters:
        algorithm_function (callable): The optimization algorithm (SA, GD, etc.).
        runs (int): Number of repetitions (bootstrapped runs).
        f (callable or None): Objective function, used for discrete evaluation.
        is_discrete (bool): Whether the problem is discrete (e.g., Ising model).
        best_state (np.array or None): Best state for discrete comparison.
        hamming_threshold (int): Distance threshold for discrete near-optimality.

    Returns:
        dict: {
            'final_values': [...],
            'final_states': [...],
            'stats': {...},
            'epsilon': float or None,
            'near_optimal_count': int or None,
            'histories': [...],
            'runtimes': [...]
        }
    """
    final_values = []
    final_states = []
    histories = []
    runtimes = []

    for i in range(1, runs + 1):
        print(f'Run {i}/{runs}...', flush=True)
        start_time = time.time()

        final_solution, _, f_history = algorithm_function(*args, **kwargs)

        duration = time.time() - start_time
        final_value = f_history[-1]

        final_values.append(final_value)
        final_states.append(final_solution)
        histories.append(f_history)
        runtimes.append(duration)

    if is_discrete:
        stats = evaluate_discrete_results(final_states, best_state, threshold=hamming_threshold)
        epsilon = None
        near_optimal_count = None
    else:
        stats = evaluate_continuous_results(final_values)
        epsilon = stats['epsilon']
        near_optimal_count = stats['near_optimal_count']

    return {
        'final_values': final_values,
        'final_states': final_states,
        'stats': stats,
        'epsilon': epsilon,
        'near_optimal_count': near_optimal_count,
        'histories': histories,
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
