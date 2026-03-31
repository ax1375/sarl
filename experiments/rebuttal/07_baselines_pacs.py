#!/usr/bin/env python3
"""Experiment 8f: SaCRL on PACS with RFF-approximated HSIC.

Uses Random Fourier Features (D=500) instead of exact HSIC to make
SaCRL tractable on full PACS data with CPU. No subsampling.

Settings:
  - Full training data (no subsampling)
  - 100 epochs
  - repr_dim = 256
  - RFF with D=500 features for HSIC approximation
  - 2 domain splits: Photo (test) and Sketch (test)
  - 3 seeds per split
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import json
from pathlib import Path

from sarl.data import MultiEnvDataset, create_multi_env_loaders
from sarl.data.pacs import get_pacs_features
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler
from sarl.training import SARLTrainer

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters ---
N_EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 126        # divisible by 3 envs
N_SEEDS = 3
DEVICE = 'cpu'
LAMBDA_INV = 5.0
REPR_DIM = 256
HIDDEN_DIMS = [256, 128]
RFF_FEATURES = 500

# Only run 2 splits for speed (Photo and Sketch as test domains)
TARGET_DOMAINS = ['photo', 'sketch']
PACS_DIR = 'C:\\Arman\\University\\Spring2026\\Research3\\data\\PACS'


def evaluate(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, *rest in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == Y).sum().item()
            total += len(Y)
    return correct / total if total > 0 else 0.0


def run_sacrl_pacs(target_domain, seed):
    """Run SaCRL on one PACS leave-one-domain-out split with RFF."""
    torch.manual_seed(seed)

    # Load features (cached from ResNet18)
    train_ds, test_ds = get_pacs_features(PACS_DIR, target_domain)
    train_loader, test_loader = create_multi_env_loaders(
        train_ds, test_ds, BATCH_SIZE, balanced=True)

    input_dim = train_ds.X.shape[1]  # 512 (ResNet18 features)
    n_envs = train_ds.n_envs  # 3 (other domains)

    model = create_sarl_model(
        'tabular', input_dim, REPR_DIM, 7, 'classification',
        encoder_kwargs={'hidden_dims': HIDDEN_DIMS})

    # Use RFF for HSIC to avoid O(n^2) kernel matrices
    objective = SARLObjective(
        lambda_inv=LAMBDA_INV, beta=1.0, rho=0.5,
        representation_dim=REPR_DIM, num_envs=n_envs,
        use_rff=True, num_features=RFF_FEATURES)

    scheduler = TemperatureScheduler(1.0, 50.0, 0.5, N_EPOCHS)
    trainer = SARLTrainer(model, objective, device=DEVICE,
                          beta_scheduler=scheduler, lr=LR)
    trainer.train(train_loader, n_epochs=N_EPOCHS, verbose=False)

    return evaluate(trainer.model, test_loader)


def main():
    print("=" * 60)
    print("PACS SaCRL WITH RFF-HSIC")
    print("=" * 60)
    print(f"Settings: epochs={N_EPOCHS}, seeds={N_SEEDS}, repr_dim={REPR_DIM}")
    print(f"RFF features={RFF_FEATURES}, full data (no subsampling)")
    print(f"Target domains: {TARGET_DOMAINS}")

    # Known baseline results
    baselines = {
        'ERM':   {'photo': 93.2, 'art_painting': 66.0, 'cartoon': 51.8, 'sketch': 37.6, 'avg': 62.2},
        'IRM':   {'photo': 93.2, 'art_painting': 66.2, 'cartoon': 51.7, 'sketch': 37.6, 'avg': 62.2},
        'VREx':  {'photo': 93.0, 'art_painting': 65.5, 'cartoon': 51.6, 'sketch': 37.9, 'avg': 62.0},
        'CIRCE': {'photo': 92.5, 'art_painting': 64.5, 'cartoon': 51.4, 'sketch': 37.3, 'avg': 61.4},
    }

    sacrl_results = {}
    for target in TARGET_DOMAINS:
        accs = []
        for s in range(N_SEEDS):
            acc = run_sacrl_pacs(target, s)
            accs.append(acc)
            print(f"  SaCRL target={target} seed={s}: {acc:.1%}", flush=True)

        t = torch.tensor(accs)
        sacrl_results[target] = {
            'accs': accs,
            'mean': t.mean().item(),
            'std': t.std().item()
        }

    print("\n" + "=" * 60)
    print("=== PACS RESULTS ===")
    print("=" * 60)

    print(f"\n| {'Method':<12} | {'Photo':<12} | {'Sketch':<12} |")
    print(f"|{'-'*14}|{'-'*14}|{'-'*14}|")
    for method, vals in baselines.items():
        p = vals.get('photo', '-')
        sk = vals.get('sketch', '-')
        print(f"| {method:<12} | {p:<12} | {sk:<12} |")

    for target in TARGET_DOMAINS:
        m = sacrl_results[target]['mean'] * 100
        s = sacrl_results[target]['std'] * 100
        print(f"SaCRL {target}: {m:.1f} +/- {s:.1f}")

    # Save
    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/pacs_sacrl_rff.json', 'w') as f:
        json.dump(sacrl_results, f, indent=2)
    print(f"\nResults saved to results/pacs_sacrl_rff.json")


if __name__ == '__main__':
    main()
