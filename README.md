# GD-SA Optimization

This repository contains the code for a comparative study of Simulated Annealing (SA) and Gradient Descent (GD), including their hybridization, applied to benchmark functions, the Ising model, and neural network training.

## Overview

- **Algorithms**:
  - Gradient Descent (GD)
  - Simulated Annealing (SA): continuous and discrete variants
  - Hybrid SA-GD: alternating descent and ascent steps, using either gradient-based ("ascent") or random perturbation-based ("unif") updates
- **Problems**:
  - Benchmark functions: Rosenbrock, Rastrigin, Ackley
  - Ising model: relaxed (continuous) and discrete versions
  - Neural network training: binary classification on the UCI wine quality dataset
- **Experiments**:
  - Baseline: fixed hyperparameters
  - Best: grid-searched optimal hyperparameters
  - All experiments are bootstrapped over 50 runs

## Structure

The main components are:

- `src/optimizers/`: core implementations of GD, SA, and Hybrid algorithms
  - also includes `gd_nn.py` and `sa_nn.py` for using GD/SA to train neural networks
- `src/problems/`: objective functions for benchmarks, Ising, and helper functions for neural networks
- `src/experiments/`: scripts to run all experiments (baseline, best, hybrid)
- `src/utils/`: utilities for experiment bootstrapping, evaluation, and plotting
- `results/`: stores all result CSVs, plots, and selected hyperparameters
- `notebooks/`: Jupyter notebooks for grid search and additional analyses

## Usage

Install dependencies:
```bash
pip install -r requirements.txt ```

Run grid search (from  notebooks):

- `notebooks/benchmarks_gridsearch.ipynb`
- `notebooks/ising_gridsearch.ipynb`


Select best hyperparameters from gridsearch:

```bash
python src/experiments/select_best_hyperparams.py
```

Run experiments (baseline and hypertuned setting - best):

```bash
# Benchmarks
python src/experiments/benchmarks_baseline.py
python src/experiments/benchmarks_best.py
python src/experiments/benchmarks_hybrid_baseline.py
python src/experiments/benchmarks_hybrid_best.py

# Ising model
python src/experiments/ising_baseline.py
python src/experiments/ising_best.py
python src/experiments/ising_hybrid_baseline.py
python src/experiments/ising_hybrid_best.py
```

## Notebooks
- `benchmarks_gridsearch.ipynb` and `ising_gridsearch.ipynb`: hyperparameter tuning via grid search
- `nn_experiments.ipynb`: train a simple neural network using GD and SA on the UCI Wine Quality dataset

## Output

- CSV summaries: `results/analytical/`
- Convergence and energy plots: `results/plots/`
- Finalized experiment outputs and plots: `results/analytical/final/` and `results/plots/final/`
- Grid search logs and selected parameters: results/gridsearch/

## Notes

- All experiments are bootstrapped with 50 random initializations
- Only SA is used for the **discrete** Ising model
- Relaxed Ising model uses flattened 10×10 lattices
- Hybrid SA-GD supports two ascent variants:
  - `"ascent"`: gradient ascent steps
  - `"unif"`: uniform random perturbation
- On the Ising model, both ascent methods share the same tuned hyperparameters from `"hybrid-unif"`
- In the neural network notebook, SA performs less stably and less accurately than GD, confirming its limited use for gradient-based learning
