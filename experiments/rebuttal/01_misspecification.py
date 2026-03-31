#!/usr/bin/env python3
"""Experiment 4: Misspecification -- Bidirectional Feedback DGP.

Uses the bidirectional DGP with varying alpha:
  alpha=0   -> reduces to G2 (causal X->Y, no feedback)
  alpha=1   -> reduces to G1 (anti-causal Y->X, full feedback)
  alpha=0.5 -> genuinely bidirectional (misspecified)

All conditions use the same DGP with same spurious features and test shift,
making OOD accuracy directly comparable across conditions.

Shows that SaCRL degrades gracefully when the true DGP violates canonical assumptions.
Uses multi-start initialization with validation-based model selection (as per paper Section 4).
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
from sarl.training import train_with_restarts

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters ---
N_TRAIN = 1000       # samples per environment (3 envs x 1000 = 3000 total)
N_VAL = 500          # validation set
N_TEST = 1000        # test set
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
BATCH_SIZE = 126     # divisible by 3
N_RESTARTS = 3
N_SEEDS = 5
DEVICE = 'cpu'


def model_fn():
    """Create a fresh SARLModel."""
    return create_sarl_model('tabular', D_FEATURES, REPR_DIM, OUTPUT_DIM, 'classification',
                             encoder_kwargs={'hidden_dims': HIDDEN_DIMS})


def evaluate_accuracy(model, loader, device=DEVICE):
    """Compute accuracy on a data loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, E in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == Y).sum().item()
            total += len(Y)
    return correct / total


def get_structure_and_weights(trainer, loader):
    """Get structure ID and weights from training-time history and post-hoc violations."""
    weight_pred = trainer.objective.get_predicted_structure()
    weights = trainer.objective.get_structure_weights()
    weights_list = [w.item() if torch.is_tensor(w) else w for w in weights]

    trainer.model.eval()
    all_phi, all_y, all_e, all_preds = [], [], [], []
    with torch.no_grad():
        for i, (X, Y, E) in enumerate(loader):
            if i >= 4: break
            X_dev = X.to(trainer.device)
            phi_x, preds = trainer.model.forward_with_representation(X_dev)
            all_phi.append(phi_x)
            all_preds.append(preds)
            all_y.append(Y.to(trainer.device))
            all_e.append(E.to(trainer.device))
    phi = torch.cat(all_phi)
    y = torch.cat(all_y)
    e = torch.cat(all_e)
    preds = torch.cat(all_preds)
    violations = trainer.objective.compute_violations_only(phi, y, e, predictions=preds)
    v_dict = {'v1': violations.v1.item(), 'v2': violations.v2.item(), 'v3': violations.v3.item()}

    return weight_pred, weights_list, v_dict


def run_sacrl(seed, alpha):
    """Run SaCRL with multi-start on bidirectional DGP with given alpha."""
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=alpha, seed=seed)
    val_data = generate_bidirectional_test_data(N_VAL, D_FEATURES, alpha=alpha, shift_magnitude=0.5, seed=seed + 500)
    test_data = generate_bidirectional_test_data(N_TEST, D_FEATURES, alpha=alpha, shift_magnitude=3.0, seed=seed + 1000)

    train_ds = MultiEnvDataset(train_data.X, train_data.Y, train_data.E)
    val_ds = MultiEnvDataset(val_data.X, val_data.Y, val_data.E)
    test_ds = MultiEnvDataset(test_data.X, test_data.Y, test_data.E)
    train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, BATCH_SIZE, balanced=True)
    _, val_loader = create_multi_env_loaders(val_ds, val_ds, min(BATCH_SIZE, N_VAL))

    trainer, history = train_with_restarts(
        model_fn, train_loader, val_loader,
        n_restarts=N_RESTARTS, n_epochs=N_EPOCHS,
        lambda_inv=LAMBDA_INV, beta_start=BETA_START, beta_end=BETA_END,
        lr=LR, rho=RHO, device=DEVICE, verbose=False)

    ood_acc = evaluate_accuracy(trainer.model, test_loader, DEVICE)
    struct_id, weights, violations = get_structure_and_weights(trainer, train_loader)

    return {
        'structure_id': struct_id,
        'ood_accuracy': ood_acc,
        'violations': violations,
        'weights': weights,
    }


def run_erm(seed, alpha):
    """Run ERM on bidirectional DGP with given alpha."""
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=alpha, seed=seed)
    test_data = generate_bidirectional_test_data(N_TEST, D_FEATURES, alpha=alpha, shift_magnitude=3.0, seed=seed + 1000)

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

    ood_acc = evaluate_accuracy(model, test_loader, DEVICE)
    return {'ood_accuracy': ood_acc}


def compute_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), t.std().item()


def main():
    print("=" * 70)
    print("MISSPECIFICATION EXPERIMENT (with multi-start + validation)")
    print("=" * 70)
    print(f"Settings: n_train={N_TRAIN}x{N_ENVS}, n_val={N_VAL}, n_test={N_TEST}, "
          f"restarts={N_RESTARTS}, epochs={N_EPOCHS}, seeds={N_SEEDS}")
    seeds = list(range(N_SEEDS))

    conditions = [
        ('alpha=0.25', 0.25),
        ('alpha=0.50 (bidir.)', 0.50),
        ('alpha=0.75', 0.75),
    ]

    all_sacrl = {}
    all_erm = {}

    for label, alpha in conditions:
        # SaCRL
        print(f"\nRunning {label} + SaCRL...")
        results = []
        for s in seeds:
            r = run_sacrl(s, alpha)
            results.append(r)
            w = r['weights']
            print(f"  seed={s}: struct=G{r['structure_id']}, OOD={r['ood_accuracy']:.1%}, "
                  f"a=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
        all_sacrl[alpha] = results

        # ERM
        print(f"Running {label} + ERM...")
        erm_results = []
        for s in seeds:
            r = run_erm(s, alpha)
            erm_results.append(r)
            print(f"  seed={s}: OOD={r['ood_accuracy']:.1%}")
        all_erm[alpha] = erm_results

    # --- Report results ---
    print("\n" + "=" * 70)
    print("=== MISSPECIFICATION EXPERIMENT RESULTS ===")
    print("=" * 70)

    print(f"\n| {'Setting':<28} | {'Structure ID':<16} | {'SaCRL OOD (%)':<17} | {'ERM OOD (%)':<17} | {'Gap':<6} |")
    print(f"|{'-'*30}|{'-'*18}|{'-'*19}|{'-'*19}|{'-'*8}|")

    for label, alpha in conditions:
        sacrl_results = all_sacrl[alpha]
        erm_results = all_erm[alpha]

        structs = Counter(r['structure_id'] for r in sacrl_results)
        mode = structs.most_common(1)[0]
        sacrl_mean, sacrl_std = compute_stats([r['ood_accuracy'] for r in sacrl_results])
        erm_mean, erm_std = compute_stats([r['ood_accuracy'] for r in erm_results])
        gap = sacrl_mean - erm_mean

        struct_str = f"G{mode[0]} ({mode[1]}/{N_SEEDS})"
        print(f"| {label:<28} | {struct_str:<16} | {sacrl_mean*100:5.1f} +/- {sacrl_std*100:4.1f}    | {erm_mean*100:5.1f} +/- {erm_std*100:4.1f}    | {gap*100:+5.1f}  |")

    # Alpha weights for main misspecified condition
    bidir_results = all_sacrl[0.50]
    a1_mean, a1_std = compute_stats([r['weights'][0] for r in bidir_results])
    a2_mean, a2_std = compute_stats([r['weights'][1] for r in bidir_results])
    a3_mean, a3_std = compute_stats([r['weights'][2] for r in bidir_results])

    print(f"\nAlpha weights under misspecification (alpha=0.5, mean across seeds):")
    print(f"  alpha_1 = {a1_mean:.2f} +/- {a1_std:.2f}")
    print(f"  alpha_2 = {a2_mean:.2f} +/- {a2_std:.2f}")
    print(f"  alpha_3 = {a3_mean:.2f} +/- {a3_std:.2f}")

    # Violation values for all conditions
    print(f"\nViolation values (mean across seeds):")
    for label, alpha in conditions:
        v1_mean, _ = compute_stats([r['violations']['v1'] for r in all_sacrl[alpha]])
        v2_mean, _ = compute_stats([r['violations']['v2'] for r in all_sacrl[alpha]])
        v3_mean, _ = compute_stats([r['violations']['v3'] for r in all_sacrl[alpha]])
        print(f"  {label}: V1={v1_mean:.4f}, V2={v2_mean:.4f}, V3={v3_mean:.4f}")

    # Save results
    results_dict = {}
    for label, alpha in conditions:
        key = f"alpha_{alpha}"
        results_dict[f'{key}_sacrl'] = [
            {'structure_id': r['structure_id'], 'ood_accuracy': r['ood_accuracy'],
             'violations': r['violations'], 'weights': r['weights']}
            for r in all_sacrl[alpha]
        ]
        results_dict[f'{key}_erm'] = [
            {'ood_accuracy': r['ood_accuracy']} for r in all_erm[alpha]
        ]

    # Summary
    summary = {}
    for label, alpha in conditions:
        s_mean, s_std = compute_stats([r['ood_accuracy'] for r in all_sacrl[alpha]])
        e_mean, e_std = compute_stats([r['ood_accuracy'] for r in all_erm[alpha]])
        key = f"alpha_{alpha}"
        summary[f'{key}_sacrl'] = f"{s_mean*100:.1f} +/- {s_std*100:.1f}"
        summary[f'{key}_erm'] = f"{e_mean*100:.1f} +/- {e_std*100:.1f}"
    results_dict['summary'] = summary

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/misspecification.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to results/misspecification.json")


if __name__ == '__main__':
    main()
