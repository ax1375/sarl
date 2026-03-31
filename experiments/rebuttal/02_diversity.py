#!/usr/bin/env python3
"""Experiment 5: Insufficient Environment Diversity (Assumption 1).

Tests how SaCRL degrades when environments become increasingly similar.
Uses the bidirectional DGP (alpha=0.5) from Experiment 4, but varies
the environment diversity parameter delta_min.

delta_min scales how different the environments are:
  delta_min=1.0 -> full diversity (default env parameters)
  delta_min=0.05 -> environments nearly identical

Shows that:
  - Structure ID becomes unstable as diversity decreases
  - OOD accuracy degrades toward ERM baseline
  - With sufficient diversity, SaCRL reliably outperforms ERM
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

from sarl.data import MultiEnvDataset, create_multi_env_loaders, SyntheticData
from sarl.models import create_sarl_model
from sarl.training import train_with_restarts
from sarl.data.synthetic import _make_bidirectional_weights, _generate_bidirectional_env

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters ---
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
ALPHA = 0.5  # bidirectional mixing
DEVICE = 'cpu'
CONFOUNDER_DIM = 3


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


def generate_diverse_bidirectional(n_samples, delta_min, seed=None):
    """Generate bidirectional training data with scaled diversity.

    delta_min scales environment-specific parameters:
      - Confounder mean shifts: [-delta_min, 0, +delta_min]
      - Noise scales: [1-0.5*delta_min, 1, 1+0.5*delta_min]
      - Spurious strengths: [0.5+delta_min, 0.5+0.5*delta_min, 0.5]

    At delta_min=1.0: shifts=[-1,0,1], noise=[0.5,1,1.5], spur=[1.5,1.0,0.5]
    At delta_min=0.0: shifts=[0,0,0], noise=[1,1,1], spur=[0.5,0.5,0.5]
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu_shifts = [-delta_min, 0.0, delta_min]
    sigma_scales = [1.0 - 0.5 * delta_min, 1.0, 1.0 + 0.5 * delta_min]
    spur_strengths = [0.5 + delta_min, 0.5 + 0.5 * delta_min, 0.5]

    W1, W2, w_xy, w_uy, w_spur = _make_bidirectional_weights(D_FEATURES, CONFOUNDER_DIM)
    samples_per_env = n_samples // N_ENVS

    X_list, Y_list, E_list = [], [], []
    for env_id in range(N_ENVS):
        mu_e = torch.tensor([mu_shifts[env_id]] + [0.0] * (CONFOUNDER_DIM - 1))
        X_env, Y_env, _ = _generate_bidirectional_env(
            samples_per_env, D_FEATURES, CONFOUNDER_DIM, ALPHA,
            mu_e, sigma_scales[env_id],
            W1, W2, w_xy, w_uy, w_spur, spur_strengths[env_id])
        X_list.append(X_env)
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))

    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list),
                         structure=0, latents={'alpha': ALPHA, 'delta_min': delta_min})


def generate_test_bidirectional(n_samples, seed=None):
    """Fixed OOD test set (same across all diversity conditions)."""
    if seed is not None:
        torch.manual_seed(seed)
    W1, W2, w_xy, w_uy, w_spur = _make_bidirectional_weights(D_FEATURES, CONFOUNDER_DIM)
    mu_test = torch.zeros(CONFOUNDER_DIM)
    X, Y, _ = _generate_bidirectional_env(
        n_samples, D_FEATURES, CONFOUNDER_DIM, ALPHA,
        mu_test, 1.0, W1, W2, w_xy, w_uy,
        w_spur, spur_strength=-0.8)
    return SyntheticData(X=X, Y=Y, E=torch.zeros(n_samples).long(),
                         structure=0, latents={'alpha': ALPHA})


def generate_val_bidirectional(n_samples, seed=None):
    """Validation set with mild shift."""
    if seed is not None:
        torch.manual_seed(seed)
    W1, W2, w_xy, w_uy, w_spur = _make_bidirectional_weights(D_FEATURES, CONFOUNDER_DIM)
    mu_val = torch.zeros(CONFOUNDER_DIM)
    X, Y, _ = _generate_bidirectional_env(
        n_samples, D_FEATURES, CONFOUNDER_DIM, ALPHA,
        mu_val, 1.0, W1, W2, w_xy, w_uy,
        w_spur, spur_strength=-0.3)
    return SyntheticData(X=X, Y=Y, E=torch.zeros(n_samples).long(),
                         structure=0, latents={'alpha': ALPHA})


def compute_diversity_rank(train_data, n_envs):
    """Compute effective rank of environment diversity matrix."""
    env_means = []
    for e in range(n_envs):
        mask = train_data.E == e
        if mask.sum() > 0:
            env_means.append(train_data.X[mask].mean(dim=0))
    if len(env_means) < 2:
        return 0
    M = torch.stack(env_means)
    M_centered = M - M.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(M_centered.float())
    threshold = 0.01 * sv[0].item() if sv[0].item() > 0 else 1e-10
    return int((sv > threshold).sum().item())


def run_sacrl(seed, delta_min):
    """Run SaCRL with multi-start on bidirectional DGP with given diversity."""
    torch.manual_seed(seed)
    train_data = generate_diverse_bidirectional(N_TRAIN * N_ENVS, delta_min, seed=seed)
    val_data = generate_val_bidirectional(N_VAL, seed=seed + 500)
    test_data = generate_test_bidirectional(N_TEST, seed=seed + 1000)

    rank = compute_diversity_rank(train_data, N_ENVS)

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
    struct_id = trainer.objective.get_predicted_structure()
    weights = trainer.objective.get_structure_weights()
    weights_list = [w.item() if torch.is_tensor(w) else w for w in weights]

    return {
        'structure_id': struct_id,
        'ood_accuracy': ood_acc,
        'rank': rank,
        'weights': weights_list,
    }


def run_erm(seed, delta_min):
    """Run ERM baseline."""
    torch.manual_seed(seed)
    train_data = generate_diverse_bidirectional(N_TRAIN * N_ENVS, delta_min, seed=seed)
    test_data = generate_test_bidirectional(N_TEST, seed=seed + 1000)

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
    print("ENVIRONMENT DIVERSITY EXPERIMENT (Assumption 1)")
    print("=" * 70)
    print(f"Settings: n_train={N_TRAIN}x{N_ENVS}, n_val={N_VAL}, n_test={N_TEST}, "
          f"restarts={N_RESTARTS}, epochs={N_EPOCHS}, seeds={N_SEEDS}, alpha={ALPHA}")

    seeds = list(range(N_SEEDS))
    diversity_levels = [
        ('Full', 1.0),
        ('Moderate', 0.5),
        ('Low', 0.2),
        ('Minimal', 0.05),
    ]

    all_sacrl = {}
    all_erm = {}

    for div_label, delta_min in diversity_levels:
        print(f"\n--- {div_label} (delta_min={delta_min}) ---")

        # SaCRL
        print(f"  SaCRL:")
        results = []
        for s in seeds:
            r = run_sacrl(s, delta_min)
            results.append(r)
            w = r['weights']
            print(f"    seed={s}: G{r['structure_id']}, OOD={r['ood_accuracy']:.1%}, "
                  f"rank={r['rank']}, a=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]")
        all_sacrl[delta_min] = results

        # ERM
        print(f"  ERM:")
        erm_results = []
        for s in seeds:
            r = run_erm(s, delta_min)
            erm_results.append(r)
            print(f"    seed={s}: OOD={r['ood_accuracy']:.1%}")
        all_erm[delta_min] = erm_results

    # --- Report ---
    print("\n" + "=" * 70)
    print("=== ENVIRONMENT DIVERSITY RESULTS ===")
    print("=" * 70)

    # Main table
    print(f"\n| {'Diversity Level':<30} | {'rank(D)':<8} | {'Structure ID':<16} | {'SaCRL OOD (%)':<17} | {'ERM OOD (%)':<17} | {'Gap':<6} |")
    print(f"|{'-'*32}|{'-'*10}|{'-'*18}|{'-'*19}|{'-'*19}|{'-'*8}|")

    for div_label, delta_min in diversity_levels:
        sacrl = all_sacrl[delta_min]
        erm = all_erm[delta_min]

        avg_rank = sum(r['rank'] for r in sacrl) / len(sacrl)
        structs = Counter(r['structure_id'] for r in sacrl)
        mode = structs.most_common(1)[0]
        struct_str = f"G{mode[0]} ({mode[1]}/{N_SEEDS})"

        sacrl_mean, sacrl_std = compute_stats([r['ood_accuracy'] for r in sacrl])
        erm_mean, erm_std = compute_stats([r['ood_accuracy'] for r in erm])
        gap = sacrl_mean - erm_mean

        print(f"| {div_label} (d_min={delta_min:<4})           | {avg_rank:<8.0f} | {struct_str:<16} | {sacrl_mean*100:5.1f} +/- {sacrl_std*100:4.1f}    | {erm_mean*100:5.1f} +/- {erm_std*100:4.1f}    | {gap*100:+5.1f}  |")

    # Structure weight analysis
    print(f"\nStructure weights by diversity level (mean +/- std):")
    for div_label, delta_min in diversity_levels:
        sacrl = all_sacrl[delta_min]
        a1_mean, a1_std = compute_stats([r['weights'][0] for r in sacrl])
        a2_mean, a2_std = compute_stats([r['weights'][1] for r in sacrl])
        a3_mean, a3_std = compute_stats([r['weights'][2] for r in sacrl])
        print(f"  {div_label} (d_min={delta_min}): "
              f"a1={a1_mean:.2f}+/-{a1_std:.2f}, "
              f"a2={a2_mean:.2f}+/-{a2_std:.2f}, "
              f"a3={a3_mean:.2f}+/-{a3_std:.2f}")

    # Save results
    results_dict = {}
    for div_label, delta_min in diversity_levels:
        key = f"delta_{delta_min}"
        sacrl = all_sacrl[delta_min]
        erm = all_erm[delta_min]
        sacrl_mean, sacrl_std = compute_stats([r['ood_accuracy'] for r in sacrl])
        erm_mean, erm_std = compute_stats([r['ood_accuracy'] for r in erm])
        avg_rank = sum(r['rank'] for r in sacrl) / len(sacrl)
        structs = Counter(r['structure_id'] for r in sacrl)

        results_dict[key] = {
            'label': div_label,
            'delta_min': delta_min,
            'avg_rank': avg_rank,
            'structure_counts': dict(structs),
            'sacrl_mean': sacrl_mean, 'sacrl_std': sacrl_std,
            'erm_mean': erm_mean, 'erm_std': erm_std,
            'gap': sacrl_mean - erm_mean,
            'sacrl_details': [
                {'structure_id': r['structure_id'], 'ood_accuracy': r['ood_accuracy'],
                 'rank': r['rank'], 'weights': r['weights']}
                for r in sacrl
            ],
            'erm_details': [{'ood_accuracy': r['ood_accuracy']} for r in erm],
        }

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/diversity.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to results/diversity.json")


if __name__ == '__main__':
    main()
