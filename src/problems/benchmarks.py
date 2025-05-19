import numpy as np

# === Vector-input benchmark functions ===

def rosenbrock(x, a=1, b=100):
    x0, x1 = x[0], x[1]
    return (a - x0)**2 + b * (x1 - x0**2)**2

def rastrigin(x, A=10):
    x0, x1 = x[0], x[1]
    return A * 2 + (x0**2 - A * np.cos(2 * np.pi * x0)) + (x1**2 - A * np.cos(2 * np.pi * x1))

def ackley(x, a=20, b=0.2, c=2*np.pi):
    x0, x1 = x[0], x[1]
    return -a * np.exp(-b * np.sqrt(0.5 * (x0**2 + x1**2))) - np.exp(0.5 * (np.cos(c * x0) + np.cos(c * x1))) + a + np.exp(1)

# === Gradients (vector-input) ===

def grad_rosenbrock(x, a=1, b=100):
    x0, x1 = x[0], x[1]
    dx = -2 * (a - x0) - 4 * b * x0 * (x1 - x0**2)
    dy = 2 * b * (x1 - x0**2)
    return np.array([dx, dy])

def grad_rastrigin(x, A=10):
    x0, x1 = x[0], x[1]
    dx = 2 * x0 + 2 * A * np.pi * np.sin(2 * np.pi * x0)
    dy = 2 * x1 + 2 * A * np.pi * np.sin(2 * np.pi * x1)
    return np.array([dx, dy])

def grad_ackley(x, a=20, b=0.2, c=2*np.pi):
    epsilon = 1e-10
    x0, x1 = x[0], x[1]
    r = np.sqrt(0.5 * (x0**2 + x1**2)) + epsilon
    dx = (a * b / r) * x0 + (c / 2) * np.sin(c * x0)
    dy = (a * b / r) * x1 + (c / 2) * np.sin(c * x1)
    return np.array([dx, dy])

# === Scalar-input legacy versions (for backward compatibility/testing) ===

def rosenbrock_scalar(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rastrigin_scalar(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def ackley_scalar(x, y, a=20, b=0.2, c=2*np.pi):
    return -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(c * x) + np.cos(c * y))) + a + np.exp(1)
