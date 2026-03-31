"""Structure-specific violation metrics using HSIC.

Paper definitions (Section 4.1):
  V1(phi) = HSIC(phi_tilde, E_tilde)  where phi_tilde, E_tilde are residuals from
             regressing phi(X) and E on Y  (anti-causal)
  V2(phi) = HSIC(Y_tilde, E_tilde')   where Y_tilde, E_tilde' are residuals from
             regressing Y and E on phi(X)  (confounded-descendant)
  V3(phi) = HSIC(phi(X), E)           unconditional  (confounded-outcome)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, NamedTuple
from ..kernels import HSIC, HSIC_RFF, GaussianKernel, DeltaKernel, EPSILON


class StructureViolation(NamedTuple):
    v1: torch.Tensor  # Anti-causal
    v2: torch.Tensor  # Confounded-descendant
    v3: torch.Tensor  # Confounded-outcome


class ViolationMetrics(nn.Module):
    """Compute structure-specific invariance violations.
    V1(phi) = HSIC(phi(X), E | Y)     # anti-causal
    V2(phi) = HSIC(Y, E | Y_hat)      # confounded-descendant (Y_hat = predictions)
    V3(phi) = HSIC(phi(X), E)         # confounded-outcome
    """
    def __init__(self, use_rff: bool = False, num_features: int = 1000, ridge_lambda: float = 1e-3,
                 representation_dim: Optional[int] = None, label_dim: int = 1, num_envs: int = 3):
        super().__init__()
        self.use_rff = use_rff
        self.num_features = num_features
        self.ridge_lambda = ridge_lambda
        self.representation_dim = representation_dim
        self.num_envs = num_envs

        # V3: phi(X) continuous + E discrete => GaussianKernel + DeltaKernel
        if use_rff and representation_dim:
            self.hsic_phi_e = HSIC_RFF(representation_dim, num_envs, num_features)
        else:
            self.hsic_phi_e = HSIC(kernel_x=GaussianKernel(), kernel_y=DeltaKernel())

        # V1 and V2: residualized (continuous) variables => GaussianKernel for both
        self.hsic_residual = HSIC(kernel_x=GaussianKernel(), kernel_y=GaussianKernel())

    def forward(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor,
                predictions: Optional[torch.Tensor] = None, normalize: bool = True) -> StructureViolation:
        if phi_x.numel() == 0 or y.numel() == 0 or e.numel() == 0:
            raise ValueError("Input tensors cannot be empty")

        if y.dim() == 1:
            y = y.unsqueeze(1).float()
        else:
            y = y.float()

        if e.dim() == 1:
            num_envs = max(self.num_envs, int(e.max().item()) + 1)
            e_onehot = torch.nn.functional.one_hot(e.long(), int(num_envs)).float()
        else:
            e_onehot = e.float()

        v1 = self._compute_v1(phi_x, y, e_onehot)
        # V2: condition on predictions (low-dim) instead of raw phi (high-dim)
        v2_cond = predictions if predictions is not None else phi_x
        v2 = self._compute_v2(v2_cond, y, e_onehot)
        v3 = self._compute_v3(phi_x, e_onehot)

        if normalize:
            v1 = torch.clamp(v1, min=0.0).sqrt()
            v2 = torch.clamp(v2, min=0.0).sqrt()
            v3 = torch.clamp(v3, min=0.0).sqrt()

        return StructureViolation(v1=v1, v2=v2, v3=v3)

    def _kernel_residual(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Residualize target w.r.t. conditioning via kernel ridge regression.

        Uses adaptive ridge parameter gamma = n^{-1/(2+d)} as recommended by
        the paper (Definition 4, kernel regression), where d is the conditioning
        variable's dimension and n is the sample size.
        """
        n = target.shape[0]
        d = conditioning.shape[1] if conditioning.dim() > 1 else 1
        # Adaptive ridge: gamma = n^{-1/(2+d)}
        gamma = n ** (-1.0 / (2 + d))
        kernel = GaussianKernel()
        K = kernel(conditioning)
        K_reg = K + gamma * torch.eye(n, device=K.device)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        try:
            alpha = torch.linalg.solve(K_reg, target)
        except (RuntimeError, torch.linalg.LinAlgError):
            alpha = torch.linalg.lstsq(K_reg, target).solution
        return target - K @ alpha

    def _compute_v1(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """V1 = HSIC(phi(X), E | Y) via residualization.

        Residualize phi(X) on Y, residualize E on Y, then compute HSIC
        on the continuous residuals with GaussianKernel for both.
        """
        phi_res = self._kernel_residual(phi_x, y)
        e_res = self._kernel_residual(e, y)
        return self.hsic_residual(phi_res, e_res)

    def _compute_v2(self, v2_cond: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """V2 = HSIC(Y, E | Y_hat) via kernel residualization.

        Uses prediction logits Y_hat (low-dim) as conditioning variable for V2,
        since Y_hat is a sufficient statistic of phi(X) for Y. This keeps the
        conditioning dimension comparable to V1 (which conditions on Y).
        """
        y_res = self._kernel_residual(y, v2_cond)
        e_res = self._kernel_residual(e, v2_cond)
        return self.hsic_residual(y_res, e_res)

    def _compute_v3(self, phi_x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """V3 = HSIC(phi(X), E) unconditional.

        phi(X) is continuous (GaussianKernel), E is discrete (DeltaKernel).
        """
        return self.hsic_phi_e(phi_x, e)

    def identify_structure(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor,
                           predictions: Optional[torch.Tensor] = None) -> Tuple[int, Dict[str, float]]:
        violations = self.forward(phi_x, y, e, predictions)
        v_tensor = torch.stack([violations.v1, violations.v2, violations.v3])
        structure_id = v_tensor.argmin().item() + 1
        return structure_id, {'v1': violations.v1.item(), 'v2': violations.v2.item(), 'v3': violations.v3.item()}


class CrossFittedViolationMetrics(ViolationMetrics):
    """ViolationMetrics with 2-fold cross-fitted residualization (SplitKCI protocol).

    Fit kernel regression on fold 1, residualize fold 2, then swap and average.
    This removes the overfitting bias from using the same data for both fitting
    and evaluation of the kernel ridge regression.
    """

    def _kernel_residual(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        n = target.shape[0]
        d = conditioning.shape[1] if conditioning.dim() > 1 else 1
        gamma = n ** (-1.0 / (2 + d))
        kernel = GaussianKernel()

        if target.dim() == 1:
            target = target.unsqueeze(1)

        # Random 2-fold split
        perm = torch.randperm(n, device=target.device)
        n1 = n // 2
        idx1, idx2 = perm[:n1], perm[n1:]

        residuals = torch.zeros_like(target)

        # Fold 1: fit on idx1, residualize idx2
        cond1, cond2 = conditioning[idx1], conditioning[idx2]
        K_11 = kernel(cond1)
        K_11_reg = K_11 + gamma * torch.eye(len(idx1), device=K_11.device)
        K_21 = kernel(cond2, cond1)
        try:
            alpha1 = torch.linalg.solve(K_11_reg, target[idx1])
        except (RuntimeError, torch.linalg.LinAlgError):
            alpha1 = torch.linalg.lstsq(K_11_reg, target[idx1]).solution
        residuals[idx2] = target[idx2] - K_21 @ alpha1

        # Fold 2: fit on idx2, residualize idx1
        K_22 = kernel(cond2)
        K_22_reg = K_22 + gamma * torch.eye(len(idx2), device=K_22.device)
        K_12 = kernel(cond1, cond2)
        try:
            alpha2 = torch.linalg.solve(K_22_reg, target[idx2])
        except (RuntimeError, torch.linalg.LinAlgError):
            alpha2 = torch.linalg.lstsq(K_22_reg, target[idx2]).solution
        residuals[idx1] = target[idx1] - K_12 @ alpha2

        return residuals

