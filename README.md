# GD-SA Optimization

This repository contains the code for a comparative study of Simulated Annealing (SA) and Gradient Descent (GD) applied to benchmark functions and the Ising model.

## Overview

- Algorithms: Gradient Descent (GD), Simulated Annealing (SA) in both continuous and discrete versions
- Problems: Rosenbrock, Rastrigin, Ackley, Ising (relaxed and discrete)
- Experiments: baseline (fixed hyperparameters) and best (grid search), repeated over 50 runs

## Structure

The main components are:
- `src/optimizers/`: core implementations of GD and SA
- `src/problems/`: benchmark and Ising objective functions
- `src/experiments/`: experiment scripts for all runs
- `src/utils/`: bootstrapping, plotting, summary utilities
- `results/`: stores CSV outputs, plots, and best hyperparameters
- `notebooks/`: optional analysis notebooks

## Usage

Install dependencies:
``` pip install -r requirements.txt ```

Run grid search:
```
python src/experiments/gridsearch/benchmarks_gridsearch.py
python src/experiments/gridsearch/ising_gridsearch.py
```

Select best hyperparameters from gridsearch:
```
python src/experiments/select_best_hyperparams.py
```

Run experiments:
```
python src/experiments/benchmarks_baseline.py
python src/experiments/ising_baseline.py
python src/experiments/benchmarks_best.py
python src/experiments/ising_best.py
```


## Output

- CSV summaries are in `results/analytical/`
- Convergence plots are in `results/plots/`
- Grid search outputs and selected parameters are in `results/gridsearch/`

## Notes

- All experiments are bootstrapped with 50 runs
- Only SA is used for the discrete Ising model
- Relaxed Ising model uses flattened 10x10 lattices
- Best hyperparameters are selected based on lowest mean value across runs
