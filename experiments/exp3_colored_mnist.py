#!/usr/bin/env python3
"""Experiment 3: Colored MNIST"""
import sys
sys.path.insert(0, '..')

import torch
import json
from pathlib import Path

from sarl.data import create_colored_mnist, create_multi_env_loaders
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler
from sarl.training import SARLTrainer


def run_colored_mnist(n_epochs=50, device='auto'):
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_ds, test_ds = create_colored_mnist('./data', [0.9, 0.8], 0.1, subsample=5000)
    train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, 128)
    
    model = create_sarl_model('image_small', in_channels=3, representation_dim=128, output_dim=2, 
                              task='classification', encoder_kwargs={'in_channels': 3})
    objective = SARLObjective(1.0, 1.0, representation_dim=128, num_envs=2)
    scheduler = TemperatureScheduler(1.0, 50.0, n_epochs//10, n_epochs)
    trainer = SARLTrainer(model, objective, device=device, beta_scheduler=scheduler, lr=1e-4)
    
    history = trainer.train(train_loader, test_loader, n_epochs=n_epochs, verbose=True)
    
    test_metrics = trainer.evaluate(test_loader)
    struct_id, violations = trainer.identify_structure(train_loader)
    
    print(f"\nResults: OOD Accuracy={test_metrics['accuracy']:.1%}, Structure=G{struct_id}")
    print(f"Violations: V1={violations['v1']:.4f}, V2={violations['v2']:.4f}, V3={violations['v3']:.4f}")
    
    return {'ood_accuracy': test_metrics['accuracy'], 'structure': struct_id, 'violations': violations}


def main():
    print("SARL Colored MNIST Experiment\n" + "="*50)
    results = run_colored_mnist(n_epochs=30)
    Path('results').mkdir(exist_ok=True)
    with open('results/colored_mnist.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
