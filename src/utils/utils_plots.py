import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def plot_convergence_curve(gd_f_history, sa_f_history, benchmark_name, experiment_id, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(gd_f_history, label="Gradient Descent")
    plt.plot(sa_f_history, label="Simulated Annealing")
    plt.xlabel("Iteration")
    plt.ylabel(f"{benchmark_name} Function Value")
    plt.title(f"Convergence on {benchmark_name} (Exp {experiment_id})")
    plt.legend()
    plt.grid(True)
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"plot_{benchmark_name}_exp{experiment_id}.png")
    plt.savefig(path)
    plt.close()
    print(f"Convergence plot saved to {path}")

def plot_3d_trajectory(f, history, benchmark_name, algorithm_label, experiment_id, plots_dir):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    traj = np.array(history)
    Z_traj = [f(pt[0], pt[1]) for pt in traj]
    ax.plot(traj[:, 0], traj[:, 1], Z_traj, color='r', marker='o', label=algorithm_label)

    ax.set_title(f"3D Trajectory on {benchmark_name} using {algorithm_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    plt.legend()
    os.makedirs(plots_dir, exist_ok=True)
    filename = f"3d_{benchmark_name}_{algorithm_label}_exp{experiment_id}.png"
    path = os.path.join(plots_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"3D trajectory plot saved to {path}")

def plot_3d_trajectory_interactive(f, history, benchmark_name, algorithm_label, experiment_id):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.7)
    traj = np.array(history)
    traj_z = [f(pt[0], pt[1]) for pt in traj]

    trajectory = go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj_z,
        mode='lines+markers', marker=dict(size=4, color='red'),
        line=dict(width=2, color='red'), name=algorithm_label
    )

    fig = go.Figure(data=[surface, trajectory])
    fig.update_layout(
        title=f"Interactive 3D Trajectory on {benchmark_name} ({algorithm_label})",
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)')
    )
    fig.show()
