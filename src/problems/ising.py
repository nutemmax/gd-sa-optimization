import numpy as np

def ising_energy(state, J=1.0):
    """
    Compute the energy of a 2D Ising model with periodic boundary conditions.
    """
    energy = 0
    N, M = state.shape
    for i in range(N):
        for j in range(M):
            energy -= J * state[i, j] * (state[i, (j+1) % M] + state[(i+1) % N, j])
    return energy

def relaxed_ising_energy(state, J=1.0, lam=5.0):
    """
    Relaxed Ising energy with soft constraints encouraging spin values in {−1, +1}.
    """
    energy = 0.0
    N, M = state.shape
    for i in range(N):
        for j in range(M):
            energy -= J * state[i, j] * (state[i, (j+1)%M] + state[(i+1)%N, j])
            energy += lam * (1 - state[i, j]**2)**2
    return energy

def grad_relaxed_ising(state, J=1.0, lam=5.0):
    """
    Gradient of the relaxed Ising energy with respect to spin values.
    """
    N, M = state.shape
    grad = np.zeros_like(state)
    for i in range(N):
        for j in range(M):
            neighbors = (
                state[i, (j+1)%M] + state[i, (j-1)%M] +
                state[(i+1)%N, j] + state[(i-1)%N, j]
            )
            coupling_term = -J * neighbors
            penalty_term = -4 * lam * state[i, j] * (1 - state[i, j]**2)
            grad[i, j] = coupling_term + penalty_term
    return grad
