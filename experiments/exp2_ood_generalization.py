#!/usr/bin/env python3
"""Experiment 2: OOD Generalization"""
import sys
sys.path.insert(0, '..')

import torch
import json
from pathlib import Path

from sarl.data import SyntheticDataGenerator, MultiEnvDataset, create_multi_env_loaders
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler
from sarl.training import SARLTrainer


def run_ood_experiment(structure=2, shift_magnitudes=[0.5, 1.0, 1.5, 2.0, 2.5], n_trials=5, device='auto'):
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
    results = []
    
    for shift in shift_magnitudes:
        train_accs, test_accs = [], []
        for trial in range(n_trials):
            seed = trial * 100
            gen = SyntheticDataGenerator(structure, 10, 3, seed)
            train_data, test_data = gen.generate_train_test(3000, 1000, shift)
            
            train_ds = MultiEnvDataset(train_data.X, train_data.Y, train_data.E)
            test_ds = MultiEnvDataset(test_data.X, test_data.Y, test_data.E)
            train_loader, test_loader = create_multi_env_loaders(train_ds, test_ds, 128)
            
            model = create_sarl_model('tabular', 10, 32, 2, 'classification')
            objective = SARLObjective(1.0, 1.0, representation_dim=32, num_envs=3)
            scheduler = TemperatureScheduler(1.0, 50.0, 10, 100)
            trainer = SARLTrainer(model, objective, device=device, beta_scheduler=scheduler, lr=1e-3)
            trainer.train(train_loader, test_loader, n_epochs=100, verbose=False)
            
            train_metrics = trainer.evaluate(train_loader)
            test_metrics = trainer.evaluate(test_loader)
            train_accs.append(train_metrics['accuracy'])
            test_accs.append(test_metrics['accuracy'])
        
        results.append({'shift': shift, 'train_acc': sum(train_accs)/len(train_accs), 
                       'test_acc': sum(test_accs)/len(test_accs)})
        print(f"Shift={shift}: train={results[-1]['train_acc']:.1%}, test={results[-1]['test_acc']:.1%}")
    
    return results


def main():
    print("SARL OOD Generalization Experiment\n" + "="*50)
    results = run_ood_experiment(n_trials=3)
    Path('results').mkdir(exist_ok=True)
    with open('results/ood_generalization.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
