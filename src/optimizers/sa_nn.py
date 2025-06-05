# src/optimizers/sa_nn.py
import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from sklearn.utils import shuffle
import warnings

# === ACTIVATION FUNCTIONS ===

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# === HELPER ===
def unflatten_weights(flat_vector, layer_sizes):
    """
    Unflatten 1D vector into list of (W, b) pairs based on layer sizes.
    """
    weights = []
    idx = 0
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i+1]
        w_size = in_size * out_size
        W = flat_vector[idx:idx+w_size].reshape((in_size, out_size))
        idx += w_size
        b = flat_vector[idx:idx+out_size]
        idx += out_size
        weights.append((W, b))
    return weights

def forward_pass(X, weights, layer_sizes, activation="relu", output="softmax"):
    """
    Perform forward pass through the network.
    """
    a = X
    for i, (W, b) in enumerate(weights[:-1]):
        z = a @ W + b
        if activation == "relu":
            a = relu(z)
        elif activation == "sigmoid":
            a = sigmoid(z)
        else:
            raise ValueError("Unsupported activation")

    # Last layer
    W, b = weights[-1]
    logits = a @ W + b
    if output == "softmax":
        return softmax(logits)
    elif output == "sigmoid":
        return sigmoid(logits)
    else:
        return logits

def compute_loss(y_pred, y_true, loss_type="cross_entropy"):
    """
    Compute loss between predicted and true labels.
    y_pred: (N, C)
    y_true: (N,) if labels, or (N, C) if one-hot
    """
    eps = 1e-9
    if loss_type == "cross_entropy":
        y_pred = np.clip(y_pred, eps, 1 - eps)
        if y_true.ndim == 1:  # convert to one-hot
            y_true = np.eye(y_pred.shape[1])[y_true]
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    elif loss_type == "mse":
        return np.mean((y_pred - y_true)**2)
    else:
        raise ValueError("Unsupported loss type")

def perturb_weights(weights, step_size=0.1):
    noise = np.random.normal(loc=0.0, scale=step_size, size=weights.shape)
    return weights + noise

# === SA-NN ===

def sa_nn(
    x_init: np.ndarray,
    layer_sizes: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    T0: float = 10.0,
    alpha: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    step_size: float = 0.1,
    perturbation_method: str = 'normal',
    bounds: List[Tuple[float, float]] = None,
    adaptive_step_size: bool = False,
    use_mini_batch: bool = False,
    batch_size: int = 32,
    verbose: bool = False,
    loss_type: str = "cross_entropy",
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    track_val_loss: bool = True,
    track_weight_norm: bool = False,
    track_val_accuracy: bool = False,
    early_stopping: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4
) -> Tuple[
    np.ndarray, List[np.ndarray], List[float],
    Optional[List[float]], Optional[List[float]], Optional[List[float]]
]:
    """
    Trains a FNN using SA.

    The optimizer perturbs a flat weight vector according to a temperature-decaying
    schedule and accepts updates based on a Metropolis criterion. supports optional
    features such as mini-batching, weight norm tracking, validation loss tracking,
    validation accuracy tracking, and early stopping.

    Training loss is always tracked. validation loss is tracked if validation data
    is provided. early stopping is based on the validation loss and only activated
    if `early_stopping=True`.

    Parameters:
        x_init (np.ndarray): initial flat weight vector.
        layer_sizes (List[int]): network architecture, e.g., [input_dim, hidden1, ..., output_dim].
        X_train (np.ndarray): training inputs.
        y_train (np.ndarray): training labels (class indices or one-hot encoded).
        T0 (float): initial temperature.
        alpha (float): temperature decay rate (multiplied at each step).
        max_iter (int): maximum number of iterations.
        tol (float): convergence tolerance based on weight change.
        step_size (float): perturbation magnitude.
        perturbation_method (str): "normal" or "uniform" noise for updates.
        bounds (List[Tuple[float, float]]): optional bounds per parameter.
        adaptive_step_size (bool): scale step size proportionally to current T.
        use_mini_batch (bool): whether to use mini-batches during training.
        batch_size (int): mini-batch size if `use_mini_batch=True`.
        verbose (bool): whether to print debug messages.
        loss_type (str): "cross_entropy" or "mse".
        X_val (np.ndarray, optional): validation inputs.
        y_val (np.ndarray, optional): validation labels.
        track_val_loss (bool): always true; validation loss is tracked if val data provided.
        track_weight_norm (bool): whether to record L2 norm of weights.
        track_val_accuracy (bool): track classification accuracy on val set (only for CE).
        early_stopping (bool): whether to enable early stopping based on val loss.
        patience (int): number of iterations without improvement before stopping.
        min_delta (float): minimum required improvement in val loss.

    returns:
        best_x (np.ndarray): best weight vector found.
        history (List[np.ndarray]): list of accepted weight vectors.
        f_history (List[float]): training loss per iteration.
        val_loss_history (List[float] or None): validation loss if val data is given.
        weight_norm_history (List[float] or None): L2 norm per iteration if tracked.
        val_accuracy_history (List[float] or None): validation accuracy if tracked.
    """
    x = x_init.copy()
    dim = len(x)
    history = [x.copy()]
    f_history = []

    if track_val_accuracy and loss_type != "cross_entropy":
        warnings.warn("Validation accuracy tracking is only supported for cross-entropy loss.")

    if use_mini_batch:
        X_train, y_train = shuffle(X_train, y_train)
        batches = [
            (X_train[i:i+batch_size], y_train[i:i+batch_size])
            for i in range(0, len(X_train), batch_size)
        ]

    weights = unflatten_weights(x, layer_sizes)
    if use_mini_batch:
        X_batch, y_batch = batches[0]
        logits = forward_pass(X_batch, weights, layer_sizes)
        f_val = compute_loss(logits, y_batch, loss_type=loss_type)
    else:
        logits = forward_pass(X_train, weights, layer_sizes)
        f_val = compute_loss(logits, y_train, loss_type=loss_type)

    f_history.append(f_val)
    best_x = x.copy()
    best_f = f_val
    T = T0
    unchanged_count = 0
    stable_threshold = 200

    val_loss_history = []
    weight_norm_history = []
    val_accuracy_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    if X_val is not None and y_val is not None:
        val_logits = forward_pass(X_val, weights, layer_sizes)
        val_loss = compute_loss(val_logits, y_val, loss_type=loss_type)
        val_loss_history.append(val_loss)
        best_val_loss = val_loss

    if track_weight_norm:
        weight_norm_history.append(np.linalg.norm(x))

    if track_val_accuracy and X_val is not None and y_val is not None and loss_type == "cross_entropy":
        preds = np.argmax(val_logits, axis=1)
        y_true = np.argmax(y_val, axis=1) if y_val.ndim == 2 else y_val
        val_acc = np.mean(preds == y_true)
        val_accuracy_history.append(val_acc)

    for it in range(max_iter):
        effective_step = step_size * (T / T0) if adaptive_step_size else step_size

        if perturbation_method == 'normal':
            x_new = x + np.random.normal(0, effective_step, size=dim)
        elif perturbation_method == 'uniform':
            x_new = x + np.random.uniform(-effective_step, effective_step, size=dim)
        else:
            raise ValueError("Unknown perturbation method")

        if bounds is not None:
            x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])

        weights_new = unflatten_weights(x_new, layer_sizes)
        if use_mini_batch:
            if it % len(batches) == 0:
                X_train, y_train = shuffle(X_train, y_train)
                batches = [
                    (X_train[i:i+batch_size], y_train[i:i+batch_size])
                    for i in range(0, len(X_train), batch_size)
                ]
            X_batch, y_batch = batches[it % len(batches)]
            logits_new = forward_pass(X_batch, weights_new, layer_sizes)
            f_new = compute_loss(logits_new, y_batch, loss_type=loss_type)
        else:
            logits_new = forward_pass(X_train, weights_new, layer_sizes)
            f_new = compute_loss(logits_new, y_train, loss_type=loss_type)

        delta = f_new - f_val
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new
            f_val = f_new
            if f_val < best_f:
                best_x = x.copy()
                best_f = f_val

        history.append(x.copy())
        f_history.append(f_val)
        T *= alpha

        if X_val is not None and y_val is not None:
            val_logits = forward_pass(X_val, weights_new, layer_sizes)
            val_loss = compute_loss(val_logits, y_val, loss_type=loss_type)
            val_loss_history.append(val_loss)

            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if verbose:
                        print(f"[{it}] New best val loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at iteration {it}")
                    break

        if track_weight_norm:
            weight_norm_history.append(np.linalg.norm(x))

        if track_val_accuracy and X_val is not None and y_val is not None and loss_type == "cross_entropy":
            preds = np.argmax(val_logits, axis=1)
            y_true = np.argmax(y_val, axis=1) if y_val.ndim == 2 else y_val
            val_acc = np.mean(preds == y_true)
            val_accuracy_history.append(val_acc)

        if np.linalg.norm(history[-1] - history[-2]) < tol:
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= stable_threshold:
            if verbose:
                print(f"Converged at iteration {it}")
            break

    return (
        best_x,
        history,
        f_history,
        val_loss_history if X_val is not None and y_val is not None else None,
        weight_norm_history if track_weight_norm else None,
        val_accuracy_history if track_val_accuracy else None
    )



