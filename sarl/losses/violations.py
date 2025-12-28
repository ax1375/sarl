"""Structure-specific violation metrics using HSIC."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, NamedTuple
from ..kernels import HSIC, HSIC_RFF, GaussianKernel, DeltaKernel


class StructureViolation(NamedTuple):
    v1: torch.Tensor  # Anti-causal
    v2: torch.Tensor  # Confounded-descendant
    v3: torch.Tensor  # Confounded-outcome


class ViolationMetrics(nn.Module):
    """Compute structure-specific invariance violations.
    V1(φ) = HSIC(φ(X), E | Y)     # anti-causal
    V2(φ) = HSIC(Y, E | φ(X))     # confounded-descendant
    V3(φ) = HSIC(φ(X), E)         # confounded-outcome
    """
    def __init__(self, use_rff: bool = False, num_features: int = 1000, ridge_lambda: float = 1e-3,
                 representation_dim: Optional[int] = None, label_dim: int = 1, num_envs: int = 3):
        super().__init__()
        self.use_rff = use_rff
        self.num_features = num_features
        self.ridge_lambda = ridge_lambda
        self.representation_dim = representation_dim
        self.num_envs = num_envs
        
        if use_rff and representation_dim:
            self.hsic_phi_e = HSIC_RFF(representation_dim, num_envs, num_features)
        else:
            self.hsic_phi_e = HSIC(kernel_x=GaussianKernel(), kernel_y=DeltaKernel())
        self.hsic_residual = HSIC(kernel_x=GaussianKernel(), kernel_y=DeltaKernel())
    
    def forward(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor, normalize: bool = True) -> StructureViolation:
        if y.dim() == 1:
            y = y.unsqueeze(1).float()
        else:
            y = y.float()
        
        if e.dim() == 1:
            num_envs = max(self.num_envs, e.max().item() + 1)
            e_onehot = torch.nn.functional.one_hot(e.long(), int(num_envs)).float()
        else:
            e_onehot = e.float()
        
        v1 = self._compute_v1(phi_x, y, e_onehot)
        v2 = self._compute_v2(phi_x, y, e_onehot)
        v3 = self._compute_v3(phi_x, e_onehot)
        
        if normalize:
            v1 = v1.sqrt() if v1 > 0 else v1
            v2 = v2.sqrt() if v2 > 0 else v2
            v3 = v3.sqrt() if v3 > 0 else v3
        
        return StructureViolation(v1=v1, v2=v2, v3=v3)
    
    def _kernel_residual(self, target: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        n = target.shape[0]
        kernel = GaussianKernel()
        K = kernel(conditioning)
        K_reg = K + self.ridge_lambda * torch.eye(n, device=K.device)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        try:
            alpha = torch.linalg.solve(K_reg, target)
        except:
            alpha = torch.linalg.lstsq(K_reg, target).solution
        return target - K @ alpha
    
    def _compute_v1(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        phi_res = self._kernel_residual(phi_x, y)
        e_res = self._kernel_residual(e, y)
        return self.hsic_residual(phi_res, e_res)
    
    def _compute_v2(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        y_res = self._kernel_residual(y, phi_x)
        e_res = self._kernel_residual(e, phi_x)
        return self.hsic_residual(y_res, e_res)
    
    def _compute_v3(self, phi_x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.hsic_phi_e(phi_x, e)
    
    def identify_structure(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        violations = self.forward(phi_x, y, e)
        v_tensor = torch.stack([violations.v1, violations.v2, violations.v3])
        structure_id = v_tensor.argmin().item() + 1
        return structure_id, {'v1': violations.v1.item(), 'v2': violations.v2.item(), 'v3': violations.v3.item()}
