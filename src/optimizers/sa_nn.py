# src/optimizers/sa_nn.py

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from src.problems.nn import forward_pass, unflatten_weights, compute_loss
from sklearn.utils import shuffle
import warnings


def perturb_weights(weights, step_size=0.1):
    noise = np.random.normal(loc=0.0, scale=step_size, size=weights.shape)
    return weights + noise

def compute_loss(weights, X, y, forward_fn, loss_fn, unflatten_fn):
    unflattened = unflatten_fn(weights)
    predictions = forward_fn(X, unflattened)
    return loss_fn(predictions, y)


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
    track_val_loss: bool = False,
    track_weight_norm: bool = False,
    track_val_accuracy: bool = False,
    early_stopping: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4
) -> Union[
    Tuple[np.ndarray, List[np.ndarray], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float], List[float], List[float]]]:
    """
    SA for training a FNN.

    This function applies SA to optimize the weights of a NN,
    with optional mini-batching, validation loss tracking, early stopping,
    weight norm monitoring, and accuracy tracking.

    Args:
        x_init (np.ndarray): Initial flat vector of weights.
        layer_sizes (List[int]): Neural network architecture (e.g., [d, h1, h2, ..., c]).
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels (either one-hot or class indices).
        T0 (float): Initial temperature.
        alpha (float): Temperature decay rate (multiplied each iteration).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for weight change magnitude (used for convergence).
        step_size (float): Standard step size for perturbations.
        perturbation_method (str): Either 'normal' or 'uniform' for perturbing weights.
        bounds (List[Tuple[float, float]]): Optional list of (min, max) bounds for each parameter.
        adaptive_step_size (bool): Whether to scale step size proportionally to T.
        use_mini_batch (bool): Whether to use mini-batches instead of full-batch training.
        batch_size (int): Size of mini-batches if `use_mini_batch=True`.
        verbose (bool): Whether to print progress messages.
        loss_type (str): Loss function to use: "cross_entropy" or "mse".
        X_val (np.ndarray, optional): Validation input data (for tracking/early stopping).
        y_val (np.ndarray, optional): Validation labels (same format as y_train).
        track_val_loss (bool): Whether to track validation loss at each iteration.
        track_weight_norm (bool): Whether to track the L2 norm of the weight vector.
        track_val_accuracy (bool): Whether to track validation accuracy (only for "cross_entropy").

    Returns:
        Tuple containing:
            - best_x (np.ndarray): Best weight vector found.
            - history (List[np.ndarray]): List of weight vectors over iterations.
            - f_history (List[float]): Training loss at each iteration.
            - val_loss_history (List[float], optional): Validation loss at each iteration.
            - weight_norm_history (List[float], optional): L2 norm of weight vector at each iteration.
            - val_accuracy_history (List[float], optional): Validation accuracy at each iteration.

    Notes:
        - Early stopping is triggered if validation loss does not improve by at least `min_delta`
        for `early_stopping_patience` iterations.
        - Validation accuracy tracking is only meaningful for classification problems using cross-entropy.
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
        logits, _ = forward_pass(X_batch, weights)
        f_val = compute_loss(logits, y_batch, loss_type=loss_type)
    else:
        logits, _ = forward_pass(X_train, weights)
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

    if track_val_loss and X_val is not None and y_val is not None:
        val_logits, _ = forward_pass(X_val, weights)
        val_loss = compute_loss(val_logits, y_val, loss_type=loss_type)
        val_loss_history.append(val_loss)
        best_val_loss = val_loss  # initialize early stopping baseline

    if track_weight_norm:
        weight_norm_history.append(np.linalg.norm(x))

    if track_val_accuracy and X_val is not None and y_val is not None and loss_type == "cross_entropy":
        preds = np.argmax(val_logits, axis=1)
        if y_val.ndim == 2:
            y_true = np.argmax(y_val, axis=1)
        else:
            y_true = y_val
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
            # optional reshuffling every full pass through the data
            if it % len(batches) == 0:
                X_train, y_train = shuffle(X_train, y_train)
                batches = [
                    (X_train[i:i + batch_size], y_train[i:i + batch_size])
                    for i in range(0, len(X_train), batch_size)
                ]
            X_batch, y_batch = batches[it % len(batches)]
            logits_new, _ = forward_pass(X_batch, weights_new)
            f_new = compute_loss(logits_new, y_batch, loss_type=loss_type)
        else:
            logits_new, _ = forward_pass(X_train, weights_new)
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

        if track_val_loss and X_val is not None and y_val is not None:
            val_logits, _ = forward_pass(X_val, weights_new)
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
            if y_val.ndim == 2:
                y_true = np.argmax(y_val, axis=1)
            else:
                y_true = y_val
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

    # exit loop
    result = [best_x, history, f_history]
    if track_val_loss:
        result.append(val_loss_history)
    if track_weight_norm:
        result.append(weight_norm_history)
    if track_val_accuracy:
        result.append(val_accuracy_history)

    return tuple(result)