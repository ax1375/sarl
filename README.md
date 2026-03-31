# SaCRL: Structure-Agnostic Causal Representation Learning

PyTorch implementation of SaCRL, which jointly identifies causal structure and learns invariant representations across multiple environments.

SaCRL considers three canonical causal structures (anti-causal, confounded-descendant, confounded-outcome), computes structure-specific HSIC-based invariance violations, and selects the best penalty via differentiable SoftMin aggregation with temperature annealing.

## Installation

```bash
pip install -e .
# Or with full dependencies (torchvision for PACS/CMNIST):
pip install -e ".[full]"
```

**Requirements:** Python 3.8+, PyTorch 1.12+, torchvision (for image experiments)

## Quick Start

```python
from sarl.data import SyntheticDataGenerator, MultiEnvDataset, create_multi_env_loaders
from sarl.models import create_sarl_model
from sarl.training import train_sarl

# Generate data from a known causal structure
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

## Project Structure

```
sarl/                        # Core method package
  data/                      # Data generation and loading
    synthetic.py             #   Canonical DGPs (G1/G2/G3) and bidirectional DGP
    datasets.py              #   ColoredMNIST, MultiEnvDataset
    pacs.py                  #   PACS dataset loader with ResNet18 feature extraction
    loaders.py               #   Balanced environment samplers
  kernels/                   # Kernel functions and independence tests
    base.py                  #   Gaussian, Laplacian, Delta kernels
    hsic.py                  #   HSIC and HSIC-RFF (Random Fourier Features)
    rff.py                   #   Random Fourier Feature maps
  losses/                    # Objective functions and penalties
    violations.py            #   V1/V2/V3 structure-specific HSIC violations
    softmin.py               #   SoftMin aggregation with temperature scheduling
    objective.py             #   Full SaCRL objective (CVaR + invariance)
    baselines.py             #   IRM, VREx, CIRCE penalty implementations
  models/                    # Neural network architectures
    encoders.py              #   MLP, CNN, ResNet encoders
    predictors.py            #   Classification/regression heads
    combined.py              #   SARLModel (encoder + predictor)
  training/                  # Training loops
    trainer.py               #   SARLTrainer with multi-restart support
    callbacks.py             #   Early stopping, logging callbacks
  utils/                     # Metrics and visualization
experiments/
  rebuttal/                  # Rebuttal experiments (see below)
  exp1_structure_identification.py   # Original paper experiments
  exp2_ood_generalization.py
  exp3_colored_mnist.py
tests/
  test_kernels.py
  test_violations.py
```

## Rebuttal Experiments

All rebuttal experiments are in `experiments/rebuttal/`. Each script is self-contained and saves results to `experiments/rebuttal/results/`.

### Setup

```bash
cd experiments/rebuttal

# For PACS experiments, download the dataset:
# Place in ../../data/PACS/ (or update PACS_DIR in the script)
# Expected structure: PACS/{art_painting,cartoon,photo,sketch}/{class_name}/*.jpg
```

### Running All Experiments

```bash
# Q2: Structure misspecification (bidirectional DGP, alpha in {0.25, 0.5, 0.75})
python 01_misspecification.py

# Q3: Environment diversity (delta_min in {1.0, 0.5, 0.2, 0.05})
python 02_diversity.py

# W2: Cross-fitted HSIC sensitivity (standard vs 2-fold SplitKCI)
python 03_crossfitting.py

# Q5: Hyperparameter sensitivity (lambda and beta sweeps)
python 04_hyperparameter.py

# W3: Baseline comparisons on bidirectional DGP (IRM, VREx, CIRCE)
python 05_baselines_bidir.py

# W3: Baseline comparisons on Colored MNIST (all 5 methods)
python 06_baselines_cmnist.py

# W3: SaCRL on PACS with RFF-HSIC (leave-one-domain-out)
python 07_baselines_pacs.py
```

### Experiment Details

| # | Script | Question | What it tests | Runtime (CPU) |
|---|--------|----------|---------------|---------------|
| 1 | `01_misspecification.py` | Q2 | SaCRL under structure misspecification (bidirectional DGP) | ~15 min |
| 2 | `02_diversity.py` | Q3 | Effect of environment diversity on structure identification | ~20 min |
| 3 | `03_crossfitting.py` | W2 | Standard vs cross-fitted HSIC residualization | ~15 min |
| 4 | `04_hyperparameter.py` | Q5 | Sensitivity to lambda_inv and beta_max | ~30 min |
| 5 | `05_baselines_bidir.py` | W3 | IRM/VREx/CIRCE on bidirectional DGP (alpha=0.5) | ~10 min |
| 6 | `06_baselines_cmnist.py` | W3 | All methods on Colored MNIST (IRM paper protocol) | ~20 min |
| 7 | `07_baselines_pacs.py` | W3 | SaCRL on PACS with RFF-approximated HSIC | ~60 min |

### Key Results

**Bidirectional DGP (alpha=0.5) -- OOD Accuracy:**

| Method | OOD Acc (%) |
|--------|------------|
| ERM | 61.2 +/- 3.8 |
| IRM | 61.3 +/- 4.2 |
| VREx | 61.4 +/- 4.4 |
| CIRCE | 66.5 +/- 2.6 |
| **SaCRL** | **73.9 +/- 4.1** |


## Run Tests

```bash
python -m pytest tests/
# Or individually:
python tests/test_kernels.py
python tests/test_violations.py
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_inv` | 5.0 | Invariance penalty weight |
| `beta_max` | 50 | SoftMin temperature (annealed from 1 to beta_max) |
| `rho` | 0.5 | CVaR quantile for robust loss |
| `N_RESTARTS` | 3-10 | Multi-start initializations for model selection |
