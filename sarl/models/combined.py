"""Combined encoder + predictor model."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .encoders import create_encoder
from .predictors import create_predictor


class SARLModel(nn.Module):
    """Combined encoder-predictor for SARL."""
    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.encoder, self.predictor = encoder, predictor
    
    @property
    def representation_dim(self) -> int:
        return self.encoder.output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encoder(x))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward_with_representation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.predictor(z)


def create_sarl_model(input_type: str, input_dim: Optional[int] = None, representation_dim: int = 64,
                      output_dim: int = 1, task: str = 'classification', encoder_kwargs: Optional[Dict] = None,
                      predictor_hidden_dims: Optional[list] = None) -> SARLModel:
    encoder = create_encoder(input_type=input_type, input_dim=input_dim, output_dim=representation_dim, **(encoder_kwargs or {}))
    predictor = create_predictor(input_dim=representation_dim, output_dim=output_dim, task=task, hidden_dims=predictor_hidden_dims)
    return SARLModel(encoder, predictor)
