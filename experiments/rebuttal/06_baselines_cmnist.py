#!/usr/bin/env python3
"""Experiment 8e: Colored MNIST following Arjovsky et al. (2019) exactly.

Key design choices matching the original IRM paper:
  - 2-channel images (digit in one channel, other zeroed)
  - Flattened to 2*28*28 = 1568 input dims
  - MLP: 2 hidden layers, 256 units each
  - Binary output (1 logit, BCEWithLogitsLoss)
  - Label noise = 0.25
  - IRM penalty weight ~10^4 (large, as in original paper)
  - ERM warm-up: 100 steps before applying penalties
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

from sarl.data import MultiEnvDataset, create_multi_env_loaders, create_colored_mnist
from sarl.models.combined import SARLModel
from sarl.models.encoders import MLPEncoder
from sarl.models.predictors import create_predictor
from sarl.losses import (SARLObjective, TemperatureScheduler,
                          IRMPenalty, VRExPenalty, CIRCEPenalty)
from sarl.training import SARLTrainer

import warnings
warnings.filterwarnings('ignore')

# --- Hyperparameters ---
N_STEPS = 501        # Match original paper
LR = 1e-3
BATCH_SIZE = 128
N_SEEDS = 5
DEVICE = 'cpu'

# Penalty weights (IRM needs to be large per original paper)
LAMBDA_IRM = 10000.0
LAMBDA_VREX = 1000.0
LAMBDA_CIRCE = 10.0
LAMBDA_SACRL = 5.0

# ERM warm-up: apply only ERM loss for first N steps
PENALTY_ANNEAL_STEPS = 100

INPUT_DIM = 2 * 28 * 28  # 1568 (2-channel images, flattened)
HIDDEN_DIMS = [256, 256]  # Match original paper
REPR_DIM = 256  # Internal representation dim


def make_model():
    """Create MLP model with binary output (1 logit)."""
    encoder = MLPEncoder(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS,
                         output_dim=REPR_DIM, batch_norm=False)
    predictor = create_predictor(input_dim=REPR_DIM, output_dim=1,
                                 task='regression')  # 1 logit output
    return SARLModel(encoder, predictor)


def make_model_multiclass():
    """Create MLP model with 2-class output for SaCRL."""
    encoder = MLPEncoder(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS,
                         output_dim=REPR_DIM, batch_norm=False)
    predictor = create_predictor(input_dim=REPR_DIM, output_dim=2,
                                 task='classification')
    return SARLModel(encoder, predictor)


def create_data(seed):
    """Create CMNIST data, flatten to 2*28*28."""
    torch.manual_seed(seed)
    train_ds, test_ds = create_colored_mnist('./data', [0.9, 0.8], 0.1)

    # Flatten: (n, 2, 28, 28) -> (n, 1568)
    train_ds.X = train_ds.X.view(len(train_ds.X), -1)
    test_ds.X = test_ds.X.view(len(test_ds.X), -1)

    return train_ds, test_ds


def evaluate_binary(model, loader, device=DEVICE):
    """Evaluate binary model (1 logit output) using BCEWithLogitsLoss threshold."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, *rest in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X).squeeze(-1)
            preds = (logits > 0).long()
            correct += (preds == Y).sum().item()
            total += len(Y)
    return correct / total if total > 0 else 0.0


def evaluate_multiclass(model, loader, device=DEVICE):
    """Evaluate multi-class model."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, *rest in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == Y).sum().item()
            total += len(Y)
    return correct / total if total > 0 else 0.0


def irm_penalty(logits, y):
    """IRMv1 penalty: ||grad_w loss(w*logits, y)||^2 at w=1."""
    scale = torch.tensor(1., device=logits.device, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(logits * scale, y.float())
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad ** 2


def train_method(method_name, seed):
    """Train a single method on CMNIST."""
    train_ds, test_ds = create_data(seed)
    train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, BATCH_SIZE)

    if method_name == 'SaCRL':
        return train_sacrl(train_ds, test_ds, train_loader, test_loader, seed)

    model = make_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    step = 0
    for epoch in range(500):  # enough epochs to hit N_STEPS
        model.train()
        for X, Y, E in train_loader:
            X, Y, E = X.to(DEVICE), Y.to(DEVICE), E.to(DEVICE)

            logits = model(X).squeeze(-1)
            y_float = Y.float()

            # Per-environment losses and penalties
            env_ids = E.long()
            unique_envs = env_ids.unique()
            env_losses = []
            env_penalties = []

            for eid in unique_envs:
                mask = env_ids == eid
                env_logits = logits[mask]
                env_y = y_float[mask]
                env_loss = nn.functional.binary_cross_entropy_with_logits(env_logits, env_y)
                env_losses.append(env_loss)

                if method_name == 'IRM':
                    env_penalties.append(irm_penalty(env_logits, y_float[mask]))
                elif method_name == 'VREx':
                    env_penalties.append(env_loss)

            mean_loss = torch.stack(env_losses).mean()

            if method_name == 'ERM':
                loss = mean_loss
            elif method_name == 'IRM':
                penalty = torch.stack(env_penalties).mean()
                penalty_weight = LAMBDA_IRM if step >= PENALTY_ANNEAL_STEPS else 1.0
                loss = mean_loss + penalty_weight * penalty
            elif method_name == 'VREx':
                variance = torch.stack(env_penalties).var()
                penalty_weight = LAMBDA_VREX if step >= PENALTY_ANNEAL_STEPS else 1.0
                loss = mean_loss + penalty_weight * variance
            elif method_name == 'CIRCE':
                phi_x, preds = model.forward_with_representation(X)
                circe = CIRCEPenalty(lambda_circe=LAMBDA_CIRCE)
                penalty = circe.compute_penalty_only(preds.squeeze(-1), Y, E, phi_x)
                penalty_weight = LAMBDA_CIRCE if step >= PENALTY_ANNEAL_STEPS else 1.0
                loss = mean_loss + penalty_weight * penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step >= N_STEPS:
                break
        if step >= N_STEPS:
            break

    return evaluate_binary(model, test_loader)


def train_sacrl(train_ds, test_ds, train_loader, test_loader, seed):
    """Train SaCRL on CMNIST."""
    torch.manual_seed(seed)
    model = make_model_multiclass()
    objective = SARLObjective(lambda_inv=LAMBDA_SACRL, beta=1.0, rho=0.5,
                              representation_dim=REPR_DIM, num_envs=2)
    scheduler = TemperatureScheduler(1.0, 50.0, 0.5, 200)
    trainer = SARLTrainer(model, objective, device=DEVICE,
                          beta_scheduler=scheduler, lr=LR)
    trainer.train(train_loader, n_epochs=200, verbose=False)
    return evaluate_multiclass(trainer.model, test_loader)


def main():
    print("=" * 60)
    print("COLORED MNIST (IRM paper protocol)")
    print("=" * 60)
    print(f"2-channel images, flattened to {INPUT_DIM}d, MLP {HIDDEN_DIMS}")
    print(f"Label noise=0.25, penalty anneal={PENALTY_ANNEAL_STEPS} steps")
    print(f"Lambda: IRM={LAMBDA_IRM}, VREx={LAMBDA_VREX}, CIRCE={LAMBDA_CIRCE}")

    methods = ['ERM', 'IRM', 'VREx', 'CIRCE', 'SaCRL']
    results = {}

    for method in methods:
        accs = []
        for s in range(N_SEEDS):
            acc = train_method(method, s)
            accs.append(acc)
            print(f"  {method} seed={s}: {acc:.1%}", flush=True)
        results[method] = accs

    print("\n" + "=" * 60)
    print("=== COLORED MNIST RESULTS ===")
    print("=" * 60)
    print(f"\n| {'Method':<12} | {'OOD Acc (%)':<18} |")
    print(f"|{'-'*14}|{'-'*20}|")
    for method in methods:
        t = torch.tensor(results[method])
        m, s = t.mean().item(), t.std().item() if len(results[method]) > 1 else 0.0
        print(f"| {method:<12} | {m*100:5.1f} +/- {s*100:4.1f}     |")

    save_dict = {}
    for method in methods:
        t = torch.tensor(results[method])
        save_dict[method] = {
            'accs': results[method],
            'mean': t.mean().item(),
            'std': t.std().item() if len(results[method]) > 1 else 0.0
        }

    Path(os.path.dirname(__file__), 'results').mkdir(exist_ok=True)
    with open('results/cmnist_baselines_v2.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nResults saved to results/cmnist_baselines_v2.json")


if __name__ == '__main__':
    main()
