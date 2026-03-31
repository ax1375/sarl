#!/usr/bin/env python3
"""Experiment 7: Hyperparameter Sensitivity (Q5 response).

Sweeps lambda_inv and beta_max across three bidirectional DGP conditions
(alpha=0.25, 0.50, 0.75) to show SaCRL is robust to hyperparameter choices.

Two sweeps:
  1. lambda sweep: {0.5, 1.0, 2.0, 5.0} with fixed beta_max=50
  2. beta_max sweep: {20, 50, 100} with fixed lambda=5.0
  (lambda=5.0, beta_max=50 is shared between both sweeps)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

from sarl.data import (MultiEnvDataset, create_multi_env_loaders,
                        generate_bidirectional_data, generate_bidirectional_test_data)
from sarl.data.synthetic import _make_bidirectional_weights, _generate_bidirectional_env, SyntheticData
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler
from sarl.training import SARLTrainer

import warnings
warnings.filterwarnings('ignore')

# --- Fixed hyperparameters ---
N_TRAIN = 1000
N_VAL = 500
N_TEST = 1000
N_ENVS = 3
D_FEATURES = 10
REPR_DIM = 32
HIDDEN_DIMS = [256, 128]
OUTPUT_DIM = 2
LR = 1e-4
N_EPOCHS = 50
RHO = 0.5
BATCH_SIZE = 126
N_RESTARTS = 3
N_SEEDS = 5
DEVICE = 'cpu'

# Conditions
ALPHAS = [0.25, 0.50, 0.75]
LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0]
BETA_VALUES = [20, 50, 100]
DEFAULT_LAMBDA = 5.0
DEFAULT_BETA = 50.0


def model_fn():
    return create_sarl_model('tabular', D_FEATURES, REPR_DIM, OUTPUT_DIM, 'classification',
                             encoder_kwargs={'hidden_dims': HIDDEN_DIMS})


def evaluate_accuracy(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, E in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == Y).sum().item()
            total += len(Y)
    return correct / total


def run_sacrl(seed, alpha, lambda_inv, beta_max):
    """Run SaCRL with specific hyperparameters."""
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=alpha, seed=seed)
    val_data = generate_bidirectional_test_data(N_VAL, D_FEATURES, alpha=alpha, seed=seed + 500)
    test_data = generate_bidirectional_test_data(N_TEST, D_FEATURES, alpha=alpha, seed=seed + 1000)

    train_ds = MultiEnvDataset(train_data.X, train_data.Y, train_data.E)
    val_ds = MultiEnvDataset(val_data.X, val_data.Y, val_data.E)
    test_ds = MultiEnvDataset(test_data.X, test_data.Y, test_data.E)
    train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, BATCH_SIZE, balanced=True)
    _, val_loader = create_multi_env_loaders(val_ds, val_ds, min(BATCH_SIZE, N_VAL))

    best_trainer = None
    best_val_loss = float('inf')

    for restart in range(N_RESTARTS):
        model = model_fn()
        objective = SARLObjective(
            lambda_inv=lambda_inv, beta=1.0, rho=RHO,
            representation_dim=REPR_DIM, num_envs=N_ENVS)
        beta_scheduler = TemperatureScheduler(1.0, beta_max, 0.5, N_EPOCHS)
        trainer = SARLTrainer(model, objective, device=DEVICE,
                              beta_scheduler=beta_scheduler, lr=LR)
        trainer.train(train_loader, n_epochs=N_EPOCHS, verbose=False)

        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_trainer = trainer

    ood_acc = evaluate_accuracy(best_trainer.model, test_loader, DEVICE)
    return ood_acc


def compute_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), t.std().item()


def main():
    print("=" * 70)
    print("HYPERPARAMETER SENSITIVITY EXPERIMENT (Q5 response)")
    print("=" * 70)
    print(f"Settings: restarts={N_RESTARTS}, epochs={N_EPOCHS}, seeds={N_SEEDS}")

    seeds = list(range(N_SEEDS))

    # Collect all (alpha, lambda, beta) combos to run
    # Lambda sweep: vary lambda with fixed beta=50
    # Beta sweep: vary beta with fixed lambda=5.0
    # Overlap at (lambda=5.0, beta=50)
    combos = set()
    for lam in LAMBDA_VALUES:
        combos.add((lam, DEFAULT_BETA))
    for beta in BETA_VALUES:
        combos.add((DEFAULT_LAMBDA, beta))

    results = {}  # key: (alpha, lambda, beta) -> list of ood_accs

    total_runs = len(ALPHAS) * len(combos) * N_SEEDS
    run_count = 0

    for alpha in ALPHAS:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")
        print(f"{'='*50}")

        for lam, beta in sorted(combos):
            key = (alpha, lam, beta)
            accs = []
            for s in seeds:
                run_count += 1
                acc = run_sacrl(s, alpha, lam, beta)
                accs.append(acc)
                print(f"  [{run_count}/{total_runs}] a={alpha}, l={lam}, b={beta}, seed={s}: OOD={acc:.1%}")
            results[key] = accs

    # --- Report ---
    print("\n" + "=" * 70)
    print("=== HYPERPARAMETER SENSITIVITY RESULTS ===")
    print("=" * 70)

    # Lambda sweep table (fixed beta=50)
    print(f"\n--- Lambda sweep (beta_max={DEFAULT_BETA:.0f} fixed) ---")
    header = f"| {'Setting':<25} |"
    for lam in LAMBDA_VALUES:
        header += f" λ={lam:<4} |"
    print(header)
    print("|" + "-" * 27 + "|" + ("|".join(["-" * 8] * len(LAMBDA_VALUES))) + "|")

    for alpha in ALPHAS:
        row = f"| α={alpha} {'':>18} |"
        for lam in LAMBDA_VALUES:
            mean, std = compute_stats(results[(alpha, lam, DEFAULT_BETA)])
            row += f" {mean*100:5.1f}  |"
        print(row)

    # Beta sweep table (fixed lambda=5.0)
    print(f"\n--- Beta_max sweep (lambda={DEFAULT_LAMBDA:.1f} fixed) ---")
    header = f"| {'Setting':<25} |"
    for beta in BETA_VALUES:
        header += f" β={beta:<4} |"
    print(header)
    print("|" + "-" * 27 + "|" + ("|".join(["-" * 8] * len(BETA_VALUES))) + "|")

    for alpha in ALPHAS:
        row = f"| α={alpha} {'':>18} |"
        for beta in BETA_VALUES:
            mean, std = compute_stats(results[(alpha, DEFAULT_LAMBDA, beta)])
            row += f" {mean*100:5.1f}  |"
        print(row)

    # Combined table for rebuttal
    print(f"\n--- Combined table (rebuttal format) ---")
    header = f"| {'Setting':<25} |"
    for lam in LAMBDA_VALUES:
        header += f" λ={lam:<4} |"
    for beta in BETA_VALUES:
        if beta == DEFAULT_BETA:
            continue  # already in lambda sweep at lambda=5.0
        header += f" β={beta:<4} |"
    print(header)

    for alpha in ALPHAS:
        row = f"| α={alpha} {'':>18} |"
        for lam in LAMBDA_VALUES:
            mean, std = compute_stats(results[(alpha, lam, DEFAULT_BETA)])
            row += f" {mean*100:5.1f}  |"
        for beta in BETA_VALUES:
            if beta == DEFAULT_BETA:
                continue
            mean, std = compute_stats(results[(alpha, DEFAULT_LAMBDA, beta)])
            row += f" {mean*100:5.1f}  |"
        print(row)

    # Detailed results with std
    print(f"\n--- Detailed (mean ± std) ---")
    for alpha in ALPHAS:
        print(f"\n  α={alpha}:")
        for lam, beta in sorted(results.keys()):
            if lam == alpha or (lam, beta) not in [(l, DEFAULT_BETA) for l in LAMBDA_VALUES] + [(DEFAULT_LAMBDA, b) for b in BETA_VALUES]:
                continue
            # Only print if alpha matches
        for key in sorted(results.keys()):
            a, lam, beta = key
            if a != alpha:
                continue
            mean, std = compute_stats(results[key])
            print(f"    λ={lam}, β={beta}: {mean*100:.1f} ± {std*100:.1f}%")

    # Save
    results_dict = {}
    for key, accs in results.items():
        alpha, lam, beta = key
        mean, std = compute_stats(accs)
        results_dict[f"a{alpha}_l{lam}_b{beta}"] = {
            'alpha': alpha, 'lambda': lam, 'beta_max': beta,
            'mean': mean, 'std': std, 'accs': accs,
        }

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/hyperparam_sensitivity.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to results/hyperparam_sensitivity.json")


if __name__ == '__main__':
    main()
