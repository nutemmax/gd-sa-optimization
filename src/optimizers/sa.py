import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from src.problems.benchmarks import * # all functions

def sa_continuous(
    f,
    x_init=None,
    T0=10.0,
    alpha=0.99,
    max_iter=10000,
    tol=1e-6,
    step_size=0.5,
    perturbation_method='normal',
    adaptive_step_size=False,
    init_range=(-5, 5),
    name=None
):
    """
    Simulated Annealing for continuous optimization problems.

    Parameters:
        f (callable): Objective function. Should accept a NumPy array `x`.
                      If the function only supports scalar inputs (e.g., f(x[0], x[1])),
                      a fallback is attempted automatically.
        x_init (np.array or None): Initial point. If None, randomly sampled from `bounds`.
        bounds (list of tuple): Box constraints for each dimension [(low1, high1), ..., (lowD, highD)].
        T0 (float): Initial temperature.
        alpha (float): Cooling rate (0 < alpha < 1).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (based on change in x).
        step_size (float): Step size for candidate generation.
        perturbation_method (str): `'normal'` or `'uniform'`.
        adaptive_step_size (bool): Whether step size decays with temperature.
        init_range (tuple): Range for init and clipping (e.g., (-2, 2)).

    Returns:
        best_state (np.array): Best solution found.
        history (list of np.array): List of visited points.
        f_history (list of float): Corresponding function values per iteration.

    Notes:
        - Works in arbitrary dimensions, provided the function f supports vector inputs.
        - Uses the Metropolis acceptance criterion with exponential cooling.
    """
    dim = len(x_init) if x_init is not None else 2
    x = np.array(x_init if x_init is not None else np.random.uniform(*init_range, size=dim), dtype=float)
    history = [x.copy()]

    try:
        f_val = f(x)
    except TypeError:
        f_val = f(x[0], x[1])
    f_history = [f_val]
    best_state = x.copy()
    best_f = f_val
    T = T0
    stable_threshold = 500
    unchanged_count = 0

    for _ in range(max_iter):
        effective_step = step_size * (T / T0) if adaptive_step_size else step_size

        if perturbation_method == 'normal':
            x_new = x + np.random.normal(0, effective_step, size=dim)
        elif perturbation_method == 'uniform':
            x_new = x + np.random.uniform(-effective_step, effective_step, size=dim)
        else:
            raise ValueError("Unknown perturbation_method: choose 'normal' or 'uniform'")

        x_new = np.clip(x_new, init_range[0], init_range[1])

        try:
            delta = f(x_new) - f(x)
        except TypeError:
            delta = f(x_new[0], x_new[1]) - f(x[0], x[1])

        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new

        try:
            current_f = f(x)
        except TypeError:
            current_f = f(x[0], x[1])

        if current_f < best_f:
            best_state = x.copy()
            best_f = current_f

        history.append(x.copy())
        f_history.append(current_f)

        T *= alpha

        if np.linalg.norm(history[-1] - history[-2]) < tol:
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= stable_threshold:
            break

    return best_state, history, f_history


def sa_discrete(f, x_init=None, lattice_size = (10,10), T_init=10, alpha=0.99, max_iter=10000, tol=1e-6, name=None):

    """
    Simulated Annealing for discrete spin systems (e.g., 2D Ising model).

    Parameters:
        f (callable): Energy function. Must accept a 2D NumPy array of ±1 spins.
        x_init (np.array or None): Initial lattice. If None, initialized randomly with ±1.
        lattice_size (tuple): Dimensions of the spin lattice (rows, cols).
        T_init (float): Initial temperature.
        alpha (float): Cooling rate (0 < alpha < 1).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (based on unchanged state).
    
    Returns:
        best_state (np.array): Spin configuration with lowest energy found.
        history (list of np.array): Sequence of spin states over iterations.
        f_history (list of float): Energy values per iteration.

    Notes:
        - Designed for 2D Ising models using single-spin flips.
        - Uses Metropolis criterion for acceptance.
        - Convergence based on repeated state detection.
    """
    
    state = x_init.copy()
    best_state = state.copy()
    best_f = f(state)
    history = [state.copy()]
    f_history = [best_f]
    T = T_init

    unchanged_count = 0
    stable_threshold = 10  # require 10 consecutive iterations with no state change
    prev_state = state.copy()

    for _ in range(max_iter):
        # generate a candidate: flip one random spin
        candidate = state.copy()
        N, M = candidate.shape
        i_rand = np.random.randint(0, N)
        j_rand = np.random.randint(0, M)
        candidate[i_rand, j_rand] *= -1  # flip the spin
        
        candidate_f = f(candidate)
        delta_E = candidate_f - f(state)
        
        # acceptance : according to the Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            state = candidate.copy()
            if candidate_f < best_f:
                best_state = candidate.copy()
                best_f = candidate_f
        
        history.append(state.copy())
        f_history.append(f(state))
        
        # cooling schedule
        T *= alpha  

        # check if state is unchanged compared to the previous iteration.
        if np.array_equal(state, prev_state):
            unchanged_count += 1
        else:
            unchanged_count = 0
        
        prev_state = state.copy()
        
        if unchanged_count >= stable_threshold:
            break
    
    return best_state, history, f_history

