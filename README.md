# SARL: Structure-Agnostic Representation Learning

PyTorch implementation for joint causal structure discovery and invariant representation learning.

## Installation
```bash
pip install -e .
# Or with full dependencies:
pip install -e ".[full]"
```

## Quick Start
```python
from sarl.data import SyntheticDataGenerator, MultiEnvDataset, create_multi_env_loaders
from sarl.models import create_sarl_model
from sarl.training import train_sarl

# Generate data
data = SyntheticDataGenerator(structure=1, n_features=10, n_envs=3).generate(3000)
dataset = MultiEnvDataset(data.X, data.Y, data.E)
train_loader, val_loader = create_multi_env_loaders(dataset, dataset, batch_size=128)

# Create and train model
model = create_sarl_model('tabular', input_dim=10, representation_dim=32, output_dim=2)
trainer, history = train_sarl(model, train_loader, val_loader, n_epochs=100)

# Identify structure
structure_id, violations = trainer.identify_structure(train_loader)
print(f"Identified structure: G{structure_id}")
```

## Run Experiments
```bash
cd experiments
python exp1_structure_identification.py  # Structure ID accuracy
python exp2_ood_generalization.py        # OOD generalization
python exp3_colored_mnist.py             # Colored MNIST benchmark
```

## Run Tests
```bash
cd tests
python test_kernels.py
python test_violations.py
```

## Project Structure
```
sarl/
├── kernels/     # HSIC, RFF, kernel functions
├── losses/      # Violations, SoftMin, SARL objective
├── models/      # Encoders, predictors
├── data/        # Synthetic generators, datasets
├── training/    # Trainer, callbacks
└── utils/       # Metrics, visualization
```
