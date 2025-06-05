import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Union
import warnings

def gd_nn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    loss_type: str = "cross_entropy",
    optimizer_type: str = "sgd",
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 100,
    tol: float = 1e-6,
    track_val_loss: bool = True,
    track_weight_norm: bool = False,
    track_val_accuracy: bool = False,
    early_stopping: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False
) -> Tuple[
    np.ndarray, List[np.ndarray], List[float],
    Optional[List[float]], Optional[List[float]], Optional[List[float]]
]:
    """
    Gradient-based training of a feedforward neural network using PyTorch optimizers.

    This function trains a neural network using SGD or Adam, with optional early stopping
    and metric tracking. It returns metrics and model weights in a structure aligned with `sa_nn`.

    Returns:
        - best_x (np.ndarray): Best flat model weights (by val loss if available).
        - weight_history (List[np.ndarray]): Flattened weights over epochs.
        - train_loss_history (List[float]): Training loss per epoch.
        - val_loss_history (List[float]): Validation loss per epoch (None if not applicable).
        - weight_norm_history (List[float]): L2 norm per epoch (None if not tracked).
        - val_accuracy_history (List[float]): Val accuracy per epoch (None if not tracked).
    """
    model.to(device)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long if loss_type == "cross_entropy" else torch.float32).to(device)
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val_torch = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_torch = torch.tensor(y_val, dtype=torch.long if loss_type == "cross_entropy" else torch.float32).to(device)

    criterion = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else nn.MSELoss()

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer type")

    weight_history, train_loss_history = [], []
    val_loss_history, weight_norm_history, val_accuracy_history = [], [], []

    best_weights = model.state_dict()
    best_val_loss = float('inf')
    patience_counter = 0
    unchanged_count = 0
    stable_threshold = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        total_loss /= len(train_loader.dataset)
        train_loss_history.append(total_loss)
        flat_weights = torch.cat([p.view(-1) for p in model.parameters()]).detach().cpu().numpy()
        weight_history.append(flat_weights)

        if track_weight_norm:
            weight_norm_history.append(np.linalg.norm(flat_weights))

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_torch)
                val_loss = criterion(val_outputs, y_val_torch).item()
                val_loss_history.append(val_loss)

                if early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_weights = model.state_dict()
                        patience_counter = 0
                        if verbose:
                            print(f"[Epoch {epoch}] New best val loss: {val_loss:.6f}")
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break

                if track_val_accuracy and loss_type == "cross_entropy":
                    preds = torch.argmax(val_outputs, dim=1)
                    y_true = y_val_torch if y_val_torch.ndim == 1 else torch.argmax(y_val_torch, dim=1)
                    acc = (preds == y_true).float().mean().item()
                    val_accuracy_history.append(acc)

        if epoch > 0 and abs(train_loss_history[-1] - train_loss_history[-2]) < tol:
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= stable_threshold:
            if verbose:
                print(f"Converged at epoch {epoch}")
            break

    best_x = torch.cat([p.view(-1) for p in model.parameters()]).detach().cpu().numpy()
    return (
        best_x,
        weight_history,
        train_loss_history,
        val_loss_history if X_val is not None and y_val is not None else None,
        weight_norm_history if track_weight_norm else None,
        val_accuracy_history if track_val_accuracy else None,
    )


def gd_nn_old(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    loss_type: str = "cross_entropy",
    optimizer_type: str = "sgd",
    lr: float = 0.01,
    batch_size: int = 32,
    epochs: int = 100,
    tol: float = 1e-6,
    track_val_loss: bool = False,
    track_weight_norm: bool = False,
    track_val_accuracy: bool = False,
    early_stopping: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False
) -> Union[
    Tuple[np.ndarray, List[np.ndarray], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float], List[float]],
    Tuple[np.ndarray, List[np.ndarray], List[float], List[float], List[float], List[float]]]:
    """
    Gradient-based training of a FNN using PyTorch optimizers.

    This function trains a NN using standard GD, SGD, or Adam,
    supporting optional mini-batching, early stopping, validation loss tracking, and 
    monitoring of weight norms and validation accuracy—following the same interface as `sa_nn`.

    Args:
        model (nn.Module): The PyTorch model to train. Must support forward() and contain initialized weights.
        X_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training labels (either one-hot or class indices).
        X_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training labels (either one-hot or class indices).
        optimizer_name (str): Which optimizer to use: "gd", "sgd", or "adam".
        lr (float): Learning rate for the optimizer.
        batch_size (int): Mini-batch size. If None, uses full-batch training.
        epochs (int): max number of training epochs.
        tol (float): Minimum weight change norm to detect convergence (not used with early stopping).
        verbose (bool): If True, prints progress logs during training.
        loss_type (str): Loss function to use, "cross_entropy" or "mse".
        X_val (np.ndarray, optional): Validation input data.
        y_val (np.ndarray, optional): Validation labels.
        track_val_loss (bool): Whether to track validation loss over epochs.
        track_weight_norm (bool): Whether to track L2 norm of model weights per epoch.
        track_val_accuracy (bool): Whether to track validation accuracy per epoch (only for cross-entropy).
        early_stopping (bool): Whether to apply early stopping based on validation loss.
        patience (int): Number of epochs to wait for validation loss improvement before stopping.
        min_delta (float): Minimum change in validation loss to reset patience.

    Returns:
        Tuple containing:
            - best_x (np.ndarray): Flattened best model weights (lowest validation loss if available).
            - weight_history (List[np.ndarray]): List of flattened weight vectors per epoch.
            - train_loss_history (List[float]): Training loss per epoch.
            - val_loss_history (List[float], optional): Validation loss per epoch (if enabled).
            - weight_norm_history (List[float], optional): L2 norm of model weights per epoch (if enabled).
            - val_accuracy_history (List[float], optional): Validation accuracy per epoch (if enabled).

    Notes:
        - All histories are aligned in length (one entry per epoch).
        - The returned `best_x` is determined based on validation loss if validation is provided,
          otherwise it's the final epoch’s weights.
    """

    model.to(device)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long if loss_type == "cross_entropy" else torch.float32).to(device)
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val_torch = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_torch = torch.tensor(y_val, dtype=torch.long if loss_type == "cross_entropy" else torch.float32).to(device)

    if loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Unsupported loss type")

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer type")

    history = []
    f_history = []
    val_loss_history = []
    weight_norm_history = []
    val_accuracy_history = []

    best_weights = model.state_dict()
    best_val_loss = float('inf')
    patience_counter = 0
    unchanged_count = 0
    stable_threshold = 10

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        f_history.append(epoch_loss)
        history.append(torch.cat([p.view(-1) for p in model.parameters()]).detach().cpu().numpy())

        if track_weight_norm:
            flat_weights = torch.cat([p.view(-1) for p in model.parameters()])
            weight_norm_history.append(torch.norm(flat_weights).item())

        if track_val_loss and X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_torch)
                val_loss = criterion(val_outputs, y_val_torch).item()
                val_loss_history.append(val_loss)

            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_weights = model.state_dict()
                    patience_counter = 0
                    if verbose:
                        print(f"[Epoch {epoch}] New best val loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        if track_val_accuracy and X_val is not None and y_val is not None and loss_type == "cross_entropy":
            with torch.no_grad():
                val_outputs = model(X_val_torch)
                preds = torch.argmax(val_outputs, dim=1)
                y_true = y_val_torch if y_val_torch.ndim == 1 else torch.argmax(y_val_torch, dim=1)
                acc = (preds == y_true).float().mean().item()
                val_accuracy_history.append(acc)

        if epoch > 0 and abs(f_history[-1] - f_history[-2]) < tol:
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= stable_threshold:
            if verbose:
                print(f"Converged at epoch {epoch}")
            break

    # return results in same structure/order as sa_nn
    best_x = torch.cat([p.view(-1) for p in model.parameters()]).detach().cpu().numpy()
    result = [best_x, history, f_history]
    if track_val_loss:
        result.append(val_loss_history)
    if track_weight_norm:
        result.append(weight_norm_history)
    if track_val_accuracy:
        result.append(val_accuracy_history)
    return tuple(result)