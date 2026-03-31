"""HSIC implementations: exact O(n²) and RFF-based O(nD)."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .base import GaussianKernel, DeltaKernel, compute_median_bandwidth, BaseKernel, EPSILON
from .rff import RandomFourierFeatures, estimate_bandwidth_from_data

# Minimum sample size for unbiased HSIC
MIN_SAMPLES_UNBIASED_HSIC = 4


class HSIC(nn.Module):
    """Exact HSIC using full kernel matrices. O(n²) complexity."""
    def __init__(self, kernel_x: Optional[BaseKernel] = None, kernel_y: Optional[BaseKernel] = None, biased: bool = True):
        super().__init__()
        self.kernel_x = kernel_x if kernel_x is not None else GaussianKernel()
        self.kernel_y = kernel_y if kernel_y is not None else GaussianKernel()
        self.biased = biased
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor, return_kernels: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute HSIC between X and Y.

        Args:
            X: Input tensor of shape (n, d_x)
            Y: Input tensor of shape (n, d_y)
            return_kernels: If True, return (hsic, K_X, K_Y)

        Returns:
            HSIC value, or tuple of (hsic, K_X, K_Y) if return_kernels=True

        Raises:
            ValueError: If inputs are empty or have mismatched sizes
        """
        if X.numel() == 0 or Y.numel() == 0:
            raise ValueError("Input tensors cannot be empty")

        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError(f"X and Y must have same number of samples. Got X: {X.shape[0]}, Y: {Y.shape[0]}")

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
        """Compute unbiased HSIC estimator.

        Args:
            K_X: Kernel matrix for X
            K_Y: Kernel matrix for Y
            n: Number of samples

        Returns:
            Unbiased HSIC estimate

        Note:
            Falls back to biased estimator for n < 4 to avoid division by zero
        """
        if n < MIN_SAMPLES_UNBIASED_HSIC:
            # Fall back to biased estimator for small samples
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
        """Compute RFF-approximated HSIC between X and Y.

        Args:
            X: Input tensor of shape (n, d_x)
            Y: Input tensor of shape (n, d_y)

        Returns:
            Approximated HSIC value

        Raises:
            ValueError: If inputs are empty or have mismatched sizes
        """
        if X.numel() == 0 or Y.numel() == 0:
            raise ValueError("Input tensors cannot be empty")

        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError(f"X and Y must have same number of samples. Got X: {X.shape[0]}, Y: {Y.shape[0]}")

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
        """Compute residuals after conditioning on Z using kernel ridge regression.

        Args:
            target: Target variable to residualize
            conditioning: Conditioning variable Z

        Returns:
            Residualized target
        """
        n = target.shape[0]
        K_Z = self.kernel_z(conditioning)
        K_reg = K_Z + self.ridge_lambda * torch.eye(n, device=K_Z.device)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        try:
            alpha = torch.linalg.solve(K_reg, target)
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            # If direct solve fails (e.g., singular matrix), use least squares
            alpha = torch.linalg.lstsq(K_reg, target).solution

        residual = target - K_Z @ alpha
        return residual.squeeze(-1) if residual.shape[-1] == 1 else residual


def hsic_test(X: torch.Tensor, Y: torch.Tensor, num_permutations: int = 100, alpha: float = 0.05, use_rff: bool = False) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Permutation test for independence using HSIC."""
    n = X.shape[0]
    hsic_module = HSIC_RFF(X.shape[1], Y.shape[1]) if use_rff else HSIC()
    hsic_obs = hsic_module(X, Y)
    null_hsics = torch.stack([hsic_module(X, Y[torch.randperm(n)]) for _ in range(num_permutations)])
    p_value = (null_hsics >= hsic_obs).float().mean()
    return hsic_obs, p_value, p_value < alpha
