"""Base kernel functions for HSIC computation."""
import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


def compute_median_bandwidth(X: torch.Tensor) -> torch.Tensor:
    """Compute median heuristic bandwidth."""
    n = X.shape[0]
    if n > 5000:
        idx = torch.randperm(n)[:5000]
        X = X[idx]
        n = 5000
    X_flat = X.reshape(n, -1).float()
    dists_sq = torch.cdist(X_flat, X_flat, p=2).pow(2)
    mask = torch.triu(torch.ones(n, n, device=X.device, dtype=torch.bool), diagonal=1)
    dists = dists_sq[mask].sqrt()
    if dists.numel() == 0:
        return torch.tensor(1.0, device=X.device)
    return torch.clamp(torch.median(dists), min=1e-6)


class BaseKernel(nn.Module, ABC):
    """Abstract base class for kernel functions."""
    @abstractmethod
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def is_characteristic(self) -> bool:
        pass


class GaussianKernel(BaseKernel):
    """Gaussian RBF kernel: k(x, y) = exp(-||x - y||² / (2σ²))"""
    def __init__(self, bandwidth: Optional[float] = None, trainable: bool = False):
        super().__init__()
        self.trainable = trainable
        self._bandwidth_fixed = bandwidth
        if trainable and bandwidth is not None:
            self.log_bandwidth = nn.Parameter(torch.tensor(bandwidth).log())
        else:
            self.register_buffer('log_bandwidth', None)
    
    @property
    def bandwidth(self) -> Optional[torch.Tensor]:
        if self.log_bandwidth is not None:
            return self.log_bandwidth.exp()
        return self._bandwidth_fixed
    
    @property
    def is_characteristic(self) -> bool:
        return True
    
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Y is None:
            Y = X
        X_flat = X.reshape(X.shape[0], -1).float()
        Y_flat = Y.reshape(Y.shape[0], -1).float()
        sigma = self.bandwidth if self.bandwidth is not None else compute_median_bandwidth(X_flat)
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=X.device, dtype=X.dtype)
        dists_sq = torch.cdist(X_flat, Y_flat, p=2).pow(2)
        return torch.exp(-dists_sq / (2 * sigma.pow(2) + 1e-8))


class LaplacianKernel(BaseKernel):
    """Laplacian kernel: k(x, y) = exp(-||x - y|| / σ)"""
    def __init__(self, bandwidth: Optional[float] = None):
        super().__init__()
        self._bandwidth = bandwidth
    
    @property
    def is_characteristic(self) -> bool:
        return True
    
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Y is None:
            Y = X
        X_flat = X.reshape(X.shape[0], -1).float()
        Y_flat = Y.reshape(Y.shape[0], -1).float()
        sigma = self._bandwidth if self._bandwidth else compute_median_bandwidth(X_flat)
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=X.device, dtype=X.dtype)
        dists = torch.cdist(X_flat, Y_flat, p=1)
        return torch.exp(-dists / (sigma + 1e-8))


class DeltaKernel(BaseKernel):
    """Delta kernel for discrete variables: k(x, y) = 1[x == y]"""
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
    
    @property
    def is_characteristic(self) -> bool:
        return True
    
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if Y is None:
            Y = X
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if Y.dim() == 1:
            Y = Y.unsqueeze(1)
        if X.shape[1] > 1:
            X_idx = X.argmax(dim=1)
            Y_idx = Y.argmax(dim=1)
        else:
            X_idx = X.squeeze(1).long()
            Y_idx = Y.squeeze(1).long()
        K = (X_idx.unsqueeze(1) == Y_idx.unsqueeze(0)).float()
        if self.normalize:
            K = K / (K.sum(dim=1, keepdim=True).sqrt() * K.sum(dim=0, keepdim=True).sqrt() + 1e-8)
        return K
