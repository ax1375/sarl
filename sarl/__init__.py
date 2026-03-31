"""
SARL: Structure-Agnostic Representation Learning
A PyTorch library for joint causal structure discovery and invariant representation learning.
"""

__version__ = "0.1.0"

from .kernels import GaussianKernel, LaplacianKernel, DeltaKernel, HSIC, HSIC_RFF, ConditionalHSIC
from .losses import ViolationMetrics, SoftMin, SARLObjective
from .models import MLPEncoder, ResNetEncoder, Predictor, SARLModel
from .training import SARLTrainer

__all__ = [
    "GaussianKernel", "LaplacianKernel", "DeltaKernel",
    "HSIC", "HSIC_RFF", "ConditionalHSIC",
    "ViolationMetrics", "SoftMin", "SARLObjective",
    "MLPEncoder", "ResNetEncoder", "Predictor", "SARLModel",
    "SARLTrainer",
]
