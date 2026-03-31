#!/usr/bin/env python3
"""Experiment 8c: IRM, VREx, CIRCE baselines on Bidirectional DGP (alpha=0.5).

Complements existing ERM (61.2%) and SaCRL (73.9%) numbers.
Uses the same data generation and model architecture as exp4_misspecification.py.
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
from sarl.models import create_sarl_model
from sarl.losses import IRMPenalty, VRExPenalty, CIRCEPenalty

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters (match exp4) ---
N_TRAIN = 1000       # samples per environment
N_TEST = 1000
N_ENVS = 3
D_FEATURES = 10
REPR_DIM = 32
HIDDEN_DIMS = [256, 128]
OUTPUT_DIM = 2
LR = 1e-4
N_EPOCHS = 50
BATCH_SIZE = 126
N_SEEDS = 5
DEVICE = 'cpu'
ALPHA = 0.5

# Penalty strengths
LAMBDA_IRM = 1.0
LAMBDA_VREX = 1.0
LAMBDA_CIRCE = 1.0


def make_model():
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


def compute_stats(values):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), t.std().item()


def train_irm(model, train_loader, n_epochs, lambda_pen, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    method = IRMPenalty(lambda_irm=lambda_pen)
    for epoch in range(n_epochs):
        model.train()
        for X, Y, E in train_loader:
            X, Y, E = X.to(device), Y.to(device), E.to(device)
            preds = model(X)
            loss = method(preds, Y, E)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def train_vrex(model, train_loader, n_epochs, lambda_pen, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    method = VRExPenalty(lambda_vrex=lambda_pen)
    for epoch in range(n_epochs):
        model.train()
        for X, Y, E in train_loader:
            X, Y, E = X.to(device), Y.to(device), E.to(device)
            preds = model(X)
            loss = method(preds, Y, E)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def train_circe(model, train_loader, n_epochs, lambda_pen, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    method = CIRCEPenalty(lambda_circe=lambda_pen)
    for epoch in range(n_epochs):
        model.train()
        for X, Y, E in train_loader:
            X, Y, E = X.to(device), Y.to(device), E.to(device)
            phi_x, preds = model.forward_with_representation(X)
            loss = method(preds, Y, E, phi_x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def run_method(seed, method_name):
    torch.manual_seed(seed)
    train_data = generate_bidirectional_data(
        N_TRAIN * N_ENVS, D_FEATURES, N_ENVS, alpha=ALPHA, seed=seed)
    test_data = generate_bidirectional_test_data(
        N_TEST, D_FEATURES, alpha=ALPHA, shift_magnitude=3.0, seed=seed + 1000)

    train_ds = MultiEnvDataset(train_data.X, train_data.Y, train_data.E)
    test_ds = MultiEnvDataset(test_data.X, test_data.Y, test_data.E)
    train_loader, test_loader = create_multi_env_loaders(
        train_ds, test_ds, BATCH_SIZE, balanced=True)

    if method_name == 'IRM':
        model = train_irm(make_model(), train_loader, N_EPOCHS, LAMBDA_IRM, DEVICE)
    elif method_name == 'VREx':
        model = train_vrex(make_model(), train_loader, N_EPOCHS, LAMBDA_VREX, DEVICE)
    elif method_name == 'CIRCE':
        model = train_circe(make_model(), train_loader, N_EPOCHS, LAMBDA_CIRCE, DEVICE)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return evaluate_accuracy(model, test_loader, DEVICE)


def main():
    print("=" * 70)
    print("BIDIRECTIONAL DGP BASELINES (alpha=0.5)")
    print("=" * 70)
    print(f"Settings: n_train={N_TRAIN}x{N_ENVS}, n_test={N_TEST}, "
          f"epochs={N_EPOCHS}, seeds={N_SEEDS}, alpha={ALPHA}")
    print(f"Lambdas: IRM={LAMBDA_IRM}, VREx={LAMBDA_VREX}, CIRCE={LAMBDA_CIRCE}")

    seeds = list(range(N_SEEDS))
    methods = ['IRM', 'VREx', 'CIRCE']
    all_results = {}

    for method in methods:
        print(f"\nRunning {method}...")
        accs = []
        for s in seeds:
            acc = run_method(s, method)
            accs.append(acc)
            print(f"  seed={s}: OOD={acc:.1%}")
        all_results[method] = accs

    # --- Report ---
    print("\n" + "=" * 70)
    print("=== BIDIRECTIONAL DGP BASELINE RESULTS (alpha=0.5) ===")
    print("=" * 70)

    print(f"\n| {'Method':<12} | {'OOD Accuracy (%)':<20} |")
    print(f"|{'-'*14}|{'-'*22}|")

    # Include known results
    known = {'ERM': '61.2', 'SaCRL': '73.9'}
    for method in methods:
        m, s = compute_stats(all_results[method])
        print(f"| {method:<12} | {m*100:5.1f} +/- {s*100:4.1f}       |")

    print(f"|{'-'*14}|{'-'*22}|")
    for name, val in known.items():
        print(f"| {name:<12} | {val:>5}   (previous)    |")

    # Save results
    save_dict = {}
    for method in methods:
        m, s = compute_stats(all_results[method])
        save_dict[method] = {
            'accs': all_results[method],
            'mean': m,
            'std': s,
        }
    save_dict['known'] = known

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/bidir_baselines.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nResults saved to results/bidir_baselines.json")


if __name__ == '__main__':
    main()
