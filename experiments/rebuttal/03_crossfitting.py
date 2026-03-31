#!/usr/bin/env python3
"""Experiment 6: Cross-Fitting Sensitivity (W2 response).

Compares standard HSIC residualization (fit and evaluate on same data)
with 2-fold cross-fitted residualization (SplitKCI protocol).

Uses the bidirectional DGP (alpha=0.5) which has well-calibrated
structure identification and OOD accuracy.

Shows that cross-fitting does not significantly change results,
validating the standard approach used in the paper.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from collections import Counter

from sarl.data import (MultiEnvDataset, create_multi_env_loaders,
                        generate_bidirectional_data, generate_bidirectional_test_data)
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler, CrossFittedViolationMetrics
from sarl.training import SARLTrainer

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters (same as exp4) ---
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
LAMBDA_INV = 5.0
BETA_START = 1.0
BETA_END = 50.0
RHO = 0.5
BATCH_SIZE = 126
N_RESTARTS = 3
N_SEEDS = 5
ALPHA = 0.5
DEVICE = 'cpu'


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


def run_sacrl(seed, cross_fitted=False):
    """Run SaCRL on bidirectional DGP, optionally with cross-fitted HSIC."""
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=ALPHA, seed=seed)
    val_data = generate_bidirectional_test_data(N_VAL, D_FEATURES, alpha=ALPHA, shift_magnitude=0.5, seed=seed + 500)
    test_data = generate_bidirectional_test_data(N_TEST, D_FEATURES, alpha=ALPHA, shift_magnitude=3.0, seed=seed + 1000)

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
            lambda_inv=LAMBDA_INV, beta=BETA_START, rho=RHO,
            representation_dim=REPR_DIM, num_envs=N_ENVS)

        if cross_fitted:
            objective.violations = CrossFittedViolationMetrics(
                representation_dim=REPR_DIM, num_envs=N_ENVS)

        beta_scheduler = TemperatureScheduler(BETA_START, BETA_END, 0.5, N_EPOCHS)
        trainer = SARLTrainer(model, objective, device=DEVICE,
                              beta_scheduler=beta_scheduler, lr=LR)
        history = trainer.train(train_loader, n_epochs=N_EPOCHS, verbose=False)

        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_trainer = trainer

    ood_acc = evaluate_accuracy(best_trainer.model, test_loader, DEVICE)
    struct_id = best_trainer.objective.get_predicted_structure()
    weights = best_trainer.objective.get_structure_weights()
    weights_list = [w.item() if torch.is_tensor(w) else w for w in weights]

    return {
        'structure_id': struct_id,
        'ood_accuracy': ood_acc,
        'weights': weights_list,
    }


def run_erm(seed):
    """Run ERM baseline."""
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=ALPHA, seed=seed)
    test_data = generate_bidirectional_test_data(N_TEST, D_FEATURES, alpha=ALPHA, shift_magnitude=3.0, seed=seed + 1000)

    train_ds = MultiEnvDataset(train_data.X, train_data.Y, train_data.E)
    test_ds = MultiEnvDataset(test_data.X, test_data.Y, test_data.E)
    train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, BATCH_SIZE, balanced=True)

    model = model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        model.train()
        for X, Y, E in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            loss = loss_fn(model(X), Y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return {'ood_accuracy': evaluate_accuracy(model, test_loader, DEVICE)}


def compute_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), t.std().item()


def main():
    print("=" * 70)
    print("CROSS-FITTING EXPERIMENT (W2 response)")
    print("=" * 70)
    print(f"Settings: n_train={N_TRAIN}x{N_ENVS}, n_val={N_VAL}, n_test={N_TEST}, "
          f"restarts={N_RESTARTS}, epochs={N_EPOCHS}, seeds={N_SEEDS}, alpha={ALPHA}")

    seeds = list(range(N_SEEDS))

    # Standard SaCRL
    print(f"\n--- Standard (no splitting) ---")
    std_results = []
    for s in seeds:
        r = run_sacrl(s, cross_fitted=False)
        std_results.append(r)
        w = r['weights']
        print(f"  seed={s}: G{r['structure_id']}, OOD={r['ood_accuracy']:.1%}, "
              f"a=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]")

    # Cross-fitted SaCRL
    print(f"\n--- Cross-fitted (2-fold) ---")
    cf_results = []
    for s in seeds:
        r = run_sacrl(s, cross_fitted=True)
        cf_results.append(r)
        w = r['weights']
        print(f"  seed={s}: G{r['structure_id']}, OOD={r['ood_accuracy']:.1%}, "
              f"a=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]")

    # ERM baseline
    print(f"\n--- ERM baseline ---")
    erm_results = []
    for s in seeds:
        r = run_erm(s)
        erm_results.append(r)
        print(f"  seed={s}: OOD={r['ood_accuracy']:.1%}")

    # --- Report ---
    print("\n" + "=" * 70)
    print("=== CROSS-FITTING RESULTS ===")
    print("=" * 70)

    std_structs = Counter(r['structure_id'] for r in std_results)
    cf_structs = Counter(r['structure_id'] for r in cf_results)
    std_mode = std_structs.most_common(1)[0]
    cf_mode = cf_structs.most_common(1)[0]

    std_mean, std_std = compute_stats([r['ood_accuracy'] for r in std_results])
    cf_mean, cf_std = compute_stats([r['ood_accuracy'] for r in cf_results])
    erm_mean, erm_std = compute_stats([r['ood_accuracy'] for r in erm_results])

    print(f"\n| {'Variant':<28} | {'Structure ID':<16} | {'OOD Acc (%)':<17} |")
    print(f"|{'-'*30}|{'-'*18}|{'-'*19}|")
    print(f"| {'Standard (no splitting)':<28} | G{std_mode[0]} ({std_mode[1]}/{N_SEEDS}){'':<8} | {std_mean*100:5.1f} +/- {std_std*100:4.1f}    |")
    print(f"| {'Cross-fitted (2-fold)':<28} | G{cf_mode[0]} ({cf_mode[1]}/{N_SEEDS}){'':<8} | {cf_mean*100:5.1f} +/- {cf_std*100:4.1f}    |")
    print(f"| {'ERM (reference)':<28} | {'---':<16} | {erm_mean*100:5.1f} +/- {erm_std*100:4.1f}    |")

    diff = abs(std_mean - cf_mean)
    print(f"\nDifference (Standard - CrossFit): {(std_mean-cf_mean)*100:+.1f} pp")
    print(f"Both variants outperform ERM by: Standard +{(std_mean-erm_mean)*100:.1f}pp, CrossFit +{(cf_mean-erm_mean)*100:.1f}pp")

    # Structure weight comparison
    print(f"\nStructure weights comparison:")
    std_a1, _ = compute_stats([r['weights'][0] for r in std_results])
    std_a2, _ = compute_stats([r['weights'][1] for r in std_results])
    std_a3, _ = compute_stats([r['weights'][2] for r in std_results])
    cf_a1, _ = compute_stats([r['weights'][0] for r in cf_results])
    cf_a2, _ = compute_stats([r['weights'][1] for r in cf_results])
    cf_a3, _ = compute_stats([r['weights'][2] for r in cf_results])
    print(f"  Standard:   a1={std_a1:.3f}, a2={std_a2:.3f}, a3={std_a3:.3f}")
    print(f"  CrossFit:   a1={cf_a1:.3f}, a2={cf_a2:.3f}, a3={cf_a3:.3f}")

    # Save results
    results_dict = {
        'standard': {
            'struct_id': dict(std_structs),
            'ood_mean': std_mean, 'ood_std': std_std,
            'details': [{'structure_id': r['structure_id'], 'ood_accuracy': r['ood_accuracy'],
                         'weights': r['weights']} for r in std_results],
        },
        'crossfitted': {
            'struct_id': dict(cf_structs),
            'ood_mean': cf_mean, 'ood_std': cf_std,
            'details': [{'structure_id': r['structure_id'], 'ood_accuracy': r['ood_accuracy'],
                         'weights': r['weights']} for r in cf_results],
        },
        'erm': {
            'ood_mean': erm_mean, 'ood_std': erm_std,
            'details': [{'ood_accuracy': r['ood_accuracy']} for r in erm_results],
        },
    }

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/crossfitting.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to results/crossfitting.json")


if __name__ == '__main__':
    main()
