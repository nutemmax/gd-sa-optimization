import numpy as np
import inspect

def sa_gd_hybrid(
    f,
    grad_f,
    lr=0.01,
    sigma=2.0,
    T0=10.0,
    max_iter=10000,
    tol=1e-6,
    x_init=None,
    init_range=None,
    name=None
):
    """
    Simplified hybrid SA-GD algorithm for unconstrained continuous optimization.

    Alternates between gradient descent and probabilistic ascent based on
    a temperature-scaled transition probability.

    Parameters:
        f (callable): Objective function. Accepts np.array `x`.
        grad_f (callable): Gradient function. Accepts np.array `x`.
        lr (float): Learning rate for both descent and ascent steps.
        sigma (float): Scaling factor for ascent steps (σ ∈ [1, 4]).
        T0 (float): Initial temperature.
        max_iter (int): Number of iterations.
        tol (float): Convergence threshold.
        x_init (np.array or None): Initial point. Random if None.
        init_range (tuple or None): Override for clipping range. If None, set based on `name`.
        name (str or None): Name of the problem (e.g. "rosenbrock") to auto-set clipping.
    
    Returns:
        x_final (np.array): Final solution.
        x_history (list of np.array): Iterates over time.
        f_history (list of float): Function values per iterate.
    """
    # determine dimension and clip range
    dim = len(x_init) if x_init is not None else 2
    if init_range is None:
        clip_range = (-2, 2) if name and name.lower() == "rosenbrock" else (-5, 5)
    else:
        clip_range = init_range

    # initialize x
    x = np.array(x_init if x_init is not None else np.random.uniform(*clip_range, size=dim), dtype=float)
    x_prev = x.copy()
    f_prev = f(x)
    f_curr = f_prev
    history = [x.copy()]
    f_history = [f_curr]

    stagnation_counter = 0
    for t in range(max_iter):
        g = grad_f(x)
        f_curr = f(x)
        delta_f = abs(f_curr - f_prev)
        T = T0 * np.log(t + 2)
        P = np.exp(-delta_f / (T * lr))
        a = np.random.rand()

        if a < P:
            x_new = x - lr * g
        else:
            x_new = x + sigma * lr * np.random.uniform(-1, 1, size=x.shape)

        x_new = np.clip(x_new, clip_range[0], clip_range[1])

        update_norm = np.linalg.norm(x_new - x)
        grad_norm = np.linalg.norm(g)

        if update_norm < tol or grad_norm < tol:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter > 10:
            history.append(x_new.copy())
            f_history.append(f(x_new))
            break

        x = x_new
        f_prev = f_curr
        history.append(x.copy())
        f_history.append(f(x))

    return x, history, f_history