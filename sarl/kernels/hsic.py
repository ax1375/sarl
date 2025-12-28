"""HSIC implementations: exact O(n²) and RFF-based O(nD)."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .base import GaussianKernel, DeltaKernel, compute_median_bandwidth, BaseKernel
from .rff import RandomFourierFeatures, estimate_bandwidth_from_data


class HSIC(nn.Module):
    """Exact HSIC using full kernel matrices. O(n²) complexity."""
    def __init__(self, kernel_x: Optional[BaseKernel] = None, kernel_y: Optional[BaseKernel] = None, biased: bool = True):
        super().__init__()
        self.kernel_x = kernel_x if kernel_x is not None else GaussianKernel()
        self.kernel_y = kernel_y if kernel_y is not None else GaussianKernel()
        self.biased = biased
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor, return_kernels: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n = X.shape[0]
        assert Y.shape[0] == n
        K_X = self.kernel_x(X)
        K_Y = self.kernel_y(Y)
        K_X_c = self._center_kernel(K_X)
        K_Y_c = self._center_kernel(K_Y)
        if self.biased:
            hsic = (K_X_c * K_Y_c).sum() / (n * n)
        else:
            hsic = self._unbiased_hsic(K_X, K_Y, n)
        return (hsic, K_X, K_Y) if return_kernels else hsic
    
    def _center_kernel(self, K: torch.Tensor) -> torch.Tensor:
        return K - K.mean(dim=1, keepdim=True) - K.mean(dim=0, keepdim=True) + K.mean()
    
    def _unbiased_hsic(self, K_X: torch.Tensor, K_Y: torch.Tensor, n: int) -> torch.Tensor:
        if n < 4:
            return (self._center_kernel(K_X) * self._center_kernel(K_Y)).sum() / (n * n)
        K_X_t, K_Y_t = K_X.clone(), K_Y.clone()
        K_X_t.fill_diagonal_(0)
        K_Y_t.fill_diagonal_(0)
        ones = torch.ones(n, device=K_X.device)
        term1 = (K_X_t * K_Y_t).sum()
        term2 = (ones @ K_X_t @ ones) * (ones @ K_Y_t @ ones) / ((n-1) * (n-2))
        term3 = 2 * (ones @ (K_X_t @ K_Y_t) @ ones) / (n-2)
        return (term1 + term2 - term3) / (n * (n-3))


class HSIC_RFF(nn.Module):
    """Scalable HSIC using Random Fourier Features. O(nD²) complexity."""
    def __init__(self, input_dim_x: int, input_dim_y: int, num_features: int = 1000, bandwidth_x: Optional[float] = None, bandwidth_y: Optional[float] = None):
        super().__init__()
        self.num_features = num_features
        self.rff_x = RandomFourierFeatures(input_dim_x, num_features, bandwidth_x)
        self.rff_y = RandomFourierFeatures(input_dim_y, num_features, bandwidth_y)
        self._bandwidth_x_set = bandwidth_x is not None
        self._bandwidth_y_set = bandwidth_y is not None
    
    def _ensure_bandwidths(self, X: torch.Tensor, Y: torch.Tensor):
        if not self._bandwidth_x_set:
            self.rff_x.set_bandwidth(estimate_bandwidth_from_data(X))
            self._bandwidth_x_set = True
        if not self._bandwidth_y_set:
            self.rff_y.set_bandwidth(estimate_bandwidth_from_data(Y))
            self._bandwidth_y_set = True
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        assert Y.shape[0] == n
        self._ensure_bandwidths(X, Y)
        psi_X = self.rff_x(X)
        psi_Y = self.rff_y(Y)
        mu_X, mu_Y = psi_X.mean(dim=0), psi_Y.mean(dim=0)
        Sigma_XY = (psi_X.T @ psi_Y) / n
        mu_outer = mu_X.unsqueeze(1) @ mu_Y.unsqueeze(0)
        diff = Sigma_XY - mu_outer
        return (diff * diff).sum()
    
    def reset_bandwidths(self):
        self._bandwidth_x_set = False
        self._bandwidth_y_set = False


class ConditionalHSIC(nn.Module):
    """Conditional HSIC: HSIC(X, Y | Z) via residualization."""
    def __init__(self, hsic_module: Optional[Union[HSIC, HSIC_RFF]] = None, ridge_lambda: float = 1e-3, kernel_z: Optional[BaseKernel] = None):
        super().__init__()
        self.hsic = hsic_module if hsic_module is not None else HSIC()
        self.ridge_lambda = ridge_lambda
        self.kernel_z = kernel_z if kernel_z is not None else GaussianKernel()
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        X_res = self._compute_residuals(X, Z)
        Y_res = self._compute_residuals(Y, Z)
        return self.hsic(X_res, Y_res)
    
    def _compute_residuals(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        n = target.shape[0]
        K_Z = self.kernel_z(conditioning)
        K_reg = K_Z + self.ridge_lambda * torch.eye(n, device=K_Z.device)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        try:
            alpha = torch.linalg.solve(K_reg, target)
        except:
            alpha = torch.linalg.lstsq(K_reg, target).solution
        return (target - K_Z @ alpha).squeeze(-1) if (target - K_Z @ alpha).shape[-1] == 1 else (target - K_Z @ alpha)


def hsic_test(X: torch.Tensor, Y: torch.Tensor, num_permutations: int = 100, alpha: float = 0.05, use_rff: bool = False) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Permutation test for independence using HSIC."""
    n = X.shape[0]
    hsic_module = HSIC_RFF(X.shape[1], Y.shape[1]) if use_rff else HSIC()
    hsic_obs = hsic_module(X, Y)
    null_hsics = torch.stack([hsic_module(X, Y[torch.randperm(n)]) for _ in range(num_permutations)])
    p_value = (null_hsics >= hsic_obs).float().mean()
    return hsic_obs, p_value, p_value < alpha
