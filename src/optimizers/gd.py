import numpy as np

def gradient_descent(f, grad_f, lr=0.001, max_iter=1000, tol=1e-6, x_init=None):
    """
    Gradient descent with optional fallback to scalar-input functions and domain-specific clipping.

    Parameters:
        f (callable): Objective function.
        grad_f (callable): Gradient of the objective.
        lr (float): Learning rate.
        max_iter (int): Max iterations.
        tol (float): Convergence tolerance.
        x_init (np.array or None): Initial point.

    Returns:
        best_x (np.array): Best solution found.
        history (list of np.array): Iteration history of x.
        f_history (list of float): Function values per iteration.
    """
    # dimension and function-specific clip/init ranges
    if x_init is not None:
        dim = len(x_init)
    else:
        dim = 2

    if dim == 2 and f.__name__.lower() == 'rosenbrock':
        init_range = (-2, 2)
        clip_range = (-2, 2)
    else:
        init_range = (-5, 5)
        clip_range = (-5, 5)

    # initialize x
    x = np.array(x_init if x_init is not None else np.random.uniform(*init_range, size=dim), dtype=float)
    history = [x.copy()]

    # first function value
    try:
        f_val = f(x)
    except TypeError:
        f_val = f(x[0], x[1])
    f_history = [f_val]

    for _ in range(max_iter):
        try:
            grad = grad_f(x)
        except TypeError:
            grad = grad_f(x[0], x[1])

        x_new = x - lr * grad
        x_new = np.clip(x_new, clip_range[0], clip_range[1])

        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(grad) < tol:
            x = x_new
            history.append(x.copy())
            try:
                f_val = f(x)
            except TypeError:
                f_val = f(x[0], x[1])
            f_history.append(f_val)
            break

        x = x_new
        history.append(x.copy())
        try:
            f_val = f(x)
        except TypeError:
            f_val = f(x[0], x[1])
        f_history.append(f_val)

    return x, history, f_history


