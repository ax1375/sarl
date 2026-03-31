"""Base kernel functions for HSIC computation."""
import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod

# Constants for numerical stability
EPSILON = 1e-6
MIN_BANDWIDTH = 1e-6
BANDWIDTH_SUBSAMPLE_SIZE = 5000


def compute_median_bandwidth(X: torch.Tensor) -> torch.Tensor:
    """Compute median heuristic bandwidth.

    Args:
        X: Input tensor of shape (n, d)

    Returns:
        Median pairwise distance, clamped to minimum value for stability

    Raises:
        ValueError: If X is empty or has invalid shape
    """
    if X.numel() == 0:
        raise ValueError("Cannot compute bandwidth for empty tensor")
    if X.dim() < 2:
        raise ValueError(f"Expected at least 2D tensor, got {X.dim()}D")

    n = X.shape[0]
    if n > BANDWIDTH_SUBSAMPLE_SIZE:
        idx = torch.randperm(n)[:BANDWIDTH_SUBSAMPLE_SIZE]
        X = X[idx]
        n = BANDWIDTH_SUBSAMPLE_SIZE
    X_flat = X.reshape(n, -1).float()
    dists_sq = torch.cdist(X_flat, X_flat, p=2).pow(2)
    mask = torch.triu(torch.ones(n, n, device=X.device, dtype=torch.bool), diagonal=1)
    dists = dists_sq[mask].sqrt()
    if dists.numel() == 0:
        return torch.tensor(1.0, device=X.device)
    return torch.clamp(torch.median(dists), min=MIN_BANDWIDTH)


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
        """Compute Gaussian kernel matrix.

        Args:
            X: Input tensor of shape (n, d)
            Y: Optional second input of shape (m, d). If None, uses X.

        Returns:
            Kernel matrix of shape (n, m)
        """
        if X.numel() == 0:
            raise ValueError("Input tensor X cannot be empty")

        if Y is None:
            Y = X
        elif Y.numel() == 0:
            raise ValueError("Input tensor Y cannot be empty")

        X_flat = X.reshape(X.shape[0], -1).float()
        Y_flat = Y.reshape(Y.shape[0], -1).float()
        sigma = self.bandwidth if self.bandwidth is not None else compute_median_bandwidth(X_flat)
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=X.device, dtype=X.dtype)
        dists_sq = torch.cdist(X_flat, Y_flat, p=2).pow(2)
        return torch.exp(-dists_sq / (2 * sigma.pow(2) + EPSILON))


class LaplacianKernel(BaseKernel):
    """Laplacian kernel: k(x, y) = exp(-||x - y|| / σ)"""
    def __init__(self, bandwidth: Optional[float] = None):
        super().__init__()
        self._bandwidth = bandwidth
    
    @property
    def is_characteristic(self) -> bool:
        return True
    
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Laplacian kernel matrix.

        Args:
            X: Input tensor of shape (n, d)
            Y: Optional second input of shape (m, d). If None, uses X.

        Returns:
            Kernel matrix of shape (n, m)
        """
        if X.numel() == 0:
            raise ValueError("Input tensor X cannot be empty")

        if Y is None:
            Y = X
        elif Y.numel() == 0:
            raise ValueError("Input tensor Y cannot be empty")

        X_flat = X.reshape(X.shape[0], -1).float()
        Y_flat = Y.reshape(Y.shape[0], -1).float()
        sigma = self._bandwidth if self._bandwidth else compute_median_bandwidth(X_flat)
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=X.device, dtype=X.dtype)
        dists = torch.cdist(X_flat, Y_flat, p=1)
        return torch.exp(-dists / (sigma + EPSILON))


class DeltaKernel(BaseKernel):
    """Delta kernel for discrete variables: k(x, y) = 1[x == y]"""
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
    
    @property
    def is_characteristic(self) -> bool:
        return True
    
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Delta kernel matrix for discrete variables.

        Args:
            X: Input tensor of shape (n,) or (n, d)
            Y: Optional second input of shape (m,) or (m, d). If None, uses X.

        Returns:
            Kernel matrix of shape (n, m)
        """
        if X.numel() == 0:
            raise ValueError("Input tensor X cannot be empty")

        if Y is None:
            Y = X
        elif Y.numel() == 0:
            raise ValueError("Input tensor Y cannot be empty")

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
            K = K / (K.sum(dim=1, keepdim=True).sqrt() * K.sum(dim=0, keepdim=True).sqrt() + EPSILON)
        return K
