"""Predictor heads for classification and regression."""
import torch
import torch.nn as nn
from typing import List, Optional


class Predictor(nn.Module):
    """Linear predictor head."""
    def __init__(self, input_dim: int, output_dim: int = 1, task: str = 'classification'):
        super().__init__()
        self.input_dim, self.output_dim, self.task = input_dim, output_dim, task
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.fc(z)
        return logits.squeeze(-1) if self.task == 'regression' and self.output_dim == 1 else logits


class MLPPredictor(nn.Module):
    """MLP predictor head."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64], output_dim: int = 1, task: str = 'classification', dropout: float = 0.0):
        super().__init__()
        self.task, self.output_dim = task, output_dim
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.network(z)
        return out.squeeze(-1) if self.task == 'regression' and self.output_dim == 1 else out


def create_predictor(input_dim: int, output_dim: int = 1, task: str = 'classification', hidden_dims: Optional[List[int]] = None) -> nn.Module:
    return Predictor(input_dim, output_dim, task) if not hidden_dims else MLPPredictor(input_dim, hidden_dims, output_dim, task)
