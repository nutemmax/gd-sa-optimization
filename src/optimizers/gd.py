import numpy as np
import inspect

def gradient_descent(f, grad_f, lr=0.001, max_iter=1000, tol=1e-6, x_init=None, init_range=(-5, 5), name = None):
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
    # determine dim
    if x_init is not None:
        dim = len(x_init)
    else:
        dim = 2
        x_init = np.random.uniform(*init_range, size=dim)

    # initialize x
    x = np.array(x_init, dtype=float)
    if dim == 10 or dim == 100  :
        clip_range = (-1,1)
    else : 
        clip_range = init_range  # use same range for clipping
    print(f"Clip range : {clip_range}")
    history = [x.copy()]
    f_history = []

    # determine input style
    f_params = inspect.signature(f).parameters
    is_vector_input = len(f_params) == 1

    grad_params = inspect.signature(grad_f).parameters
    is_grad_vector_input = len(grad_params) == 1

    # unpack
    def call_f(x):
        # return f(x) if is_vector_input else f(*x)
        return f(np.asarray(x).flatten())

    def call_grad(x):
        # return grad_f(x) if is_grad_vector_input else np.array(grad_f(*x))
        return grad_f(np.asarray(x).flatten())
    
    f_val = call_f(x)
    f_history.append(f_val)

    for _ in range(max_iter):
        grad = call_grad(x)
        x_new = x - lr * grad
        x_new = np.clip(x_new, clip_range[0], clip_range[1])

        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(grad) < tol:
            x = x_new
            history.append(x.copy())
            f_val = call_f(x)
            f_history.append(f_val)
            break

        x = x_new
        history.append(x.copy())
        f_val = call_f(x)
        f_history.append(f_val)

    return x, history, f_history


