#!/usr/bin/env python3
"""Experiment 1: Structure Identification Accuracy"""
import sys
sys.path.insert(0, '..')

import torch
import json
from pathlib import Path
from tqdm import tqdm

from sarl.data import SyntheticDataGenerator, MultiEnvDataset, create_multi_env_loaders
from sarl.models import create_sarl_model
from sarl.losses import SARLObjective, TemperatureScheduler
from sarl.training import SARLTrainer


def run_experiment(structures=[1,2,3], sample_sizes=[500,1000,2000,5000], n_trials=10, n_epochs=100, device='auto'):
    results = {}
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for structure in structures:
        print(f"\n{'='*50}\nTesting Structure G{structure}\n{'='*50}")
        results[structure] = {}
        
        for n_samples in sample_sizes:
            correct, predictions = 0, []
            for trial in tqdm(range(n_trials), desc=f"n={n_samples}"):
                seed = trial * 1000 + structure * 100 + n_samples
                data = SyntheticDataGenerator(structure, 10, 3, seed).generate(n_samples)
                dataset = MultiEnvDataset(data.X, data.Y, data.E)
                train_loader, _ = create_multi_env_loaders(dataset, dataset, min(128, n_samples//3), True)
                
                model = create_sarl_model('tabular', 10, 32, 2, 'classification')
                objective = SARLObjective(1.0, 1.0, representation_dim=32, num_envs=3)
                scheduler = TemperatureScheduler(1.0, 50.0, 10, n_epochs)
                trainer = SARLTrainer(model, objective, device=device, beta_scheduler=scheduler, lr=1e-3)
                trainer.train(train_loader, n_epochs=n_epochs, verbose=False)
                
                pred, _ = trainer.identify_structure(train_loader)
                predictions.append(pred)
                correct += pred == structure
            
            results[structure][n_samples] = {'accuracy': correct/n_trials, 'predictions': predictions}
            print(f"  n={n_samples}: accuracy={correct/n_trials:.1%}")
    
    return results


def main():
    print("SARL Structure Identification Experiment\n" + "="*50)
    results = run_experiment(n_trials=5, n_epochs=50)  # Reduced for quick testing
    
    print("\n" + "="*50 + "\nResults Summary\n" + "="*50)
    print(f"{'Size':<10} G1      G2      G3")
    for n in [500, 1000, 2000, 5000]:
        if n in results[1]:
            print(f"{n:<10} {results[1][n]['accuracy']:.1%}   {results[2][n]['accuracy']:.1%}   {results[3][n]['accuracy']:.1%}")
    
    Path('results').mkdir(exist_ok=True)
    with open('results/structure_id.json', 'w') as f:
        json.dump({str(k): {str(n): v for n, v in sv.items()} for k, sv in results.items()}, f, indent=2)


if __name__ == '__main__':
    main()
