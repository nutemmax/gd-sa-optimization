import numpy as np

def init_weights(layer_sizes):
    """
    Initialize weights and biases for each layer.
    Returns a flat vector of all parameters.
    """
    weights = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
        b = np.zeros(layer_sizes[i+1])
        weights.append((w, b))
    return flatten_weights(weights)

def flatten_weights(weight_list):
    """
    Flatten list of (W, b) pairs into a single 1D vector.
    """
    flat = [w.flatten() for w, b in weight_list] + [b.flatten() for w, b in weight_list]
    return np.concatenate(flat)

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

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
