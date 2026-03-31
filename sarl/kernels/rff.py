"""Random Fourier Features for scalable kernel approximation."""
import torch
import torch.nn as nn
import math
from typing import Optional


class RandomFourierFeatures(nn.Module):
    """RFF map for Gaussian RBF kernel approximation."""
    def __init__(self, input_dim: int, num_features: int = 1000, bandwidth: Optional[float] = None, learnable: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.learnable = learnable
        self.register_buffer('omega_unscaled', torch.randn(input_dim, num_features))
        self.register_buffer('b', torch.rand(num_features) * 2 * math.pi)
        if learnable and bandwidth is not None:
            self.log_bandwidth = nn.Parameter(torch.tensor(float(bandwidth)).log())
        elif bandwidth is not None:
            self.register_buffer('log_bandwidth', torch.tensor(float(bandwidth)).log())
        else:
            self.register_buffer('log_bandwidth', None)
    
    @property
    def bandwidth(self) -> Optional[torch.Tensor]:
        return self.log_bandwidth.exp() if self.log_bandwidth is not None else None
    
    def set_bandwidth(self, bandwidth: float):
        device = self.omega_unscaled.device
        if self.learnable:
            self.log_bandwidth.data = torch.tensor(float(bandwidth), device=device).log()
        else:
            self.register_buffer('log_bandwidth', torch.tensor(float(bandwidth), device=device).log())
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_flat = X.reshape(X.shape[0], -1).float()
        if X_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {X_flat.shape[1]}")
        if self.bandwidth is None:
            raise ValueError("Bandwidth not set. Call set_bandwidth() first.")
        omega = self.omega_unscaled / self.bandwidth
        projection = X_flat @ omega + self.b.unsqueeze(0)
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


def estimate_bandwidth_from_data(X: torch.Tensor, percentile: float = 50.0) -> float:
    """Estimate bandwidth using percentile of pairwise distances."""
    X_flat = X.reshape(X.shape[0], -1).float()
    n = X_flat.shape[0]
    if n > 5000:
        X_flat = X_flat[torch.randperm(n)[:5000]]
    dists = torch.cdist(X_flat, X_flat, p=2)
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    pairwise_dists = dists[mask]
    if pairwise_dists.numel() == 0:
        return 1.0
    return max(float(torch.quantile(pairwise_dists, percentile / 100.0)), 1e-6)
