"""SARL objective: L(φ, w) = CVaR_ρ({R_e}) + λ · SoftMin_β({V_k})"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from .violations import ViolationMetrics, StructureViolation
from .softmin import SoftMin, AdaptiveWeights


class CVaR(nn.Module):
    """Conditional Value-at-Risk for distributionally robust optimization."""
    def __init__(self, rho: float = 0.5):
        super().__init__()
        assert 0 < rho <= 1
        self.rho = rho
    
    def forward(self, risks: Union[torch.Tensor, List[torch.Tensor]], return_weights: bool = False):
        if isinstance(risks, list):
            risks = torch.stack(risks)
        num_envs = risks.numel()
        if self.rho >= 1.0:
            cvar = risks.max()
            weights = (risks == cvar).float() / (risks == cvar).sum()
        else:
            sorted_risks, indices = torch.sort(risks, descending=True)
            k = max(1, int(self.rho * num_envs))
            cvar = sorted_risks[:k].mean()
            weights = torch.zeros_like(risks)
            weights[indices[:k]] = 1.0 / k
        return (cvar, weights) if return_weights else cvar


class SARLObjective(nn.Module):
    """Structure-Agnostic Representation Learning objective."""
    def __init__(self, lambda_inv: float = 1.0, beta: float = 10.0, rho: float = 0.5, use_rff: bool = False,
                 num_features: int = 1000, representation_dim: Optional[int] = None, num_envs: int = 3, label_dim: int = 1):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.cvar = CVaR(rho=rho)
        self.adaptive_weights = AdaptiveWeights(beta=beta, num_structures=3)
        self.violations = ViolationMetrics(use_rff=use_rff, num_features=num_features, representation_dim=representation_dim, num_envs=num_envs)
    
    @property
    def beta(self) -> torch.Tensor:
        return self.adaptive_weights.beta
    
    def set_beta(self, beta: float):
        self.adaptive_weights.set_beta(beta)
    
    def forward(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor, predictions: torch.Tensor,
                loss_fn: nn.Module = None, return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='none') if y.dtype in [torch.long, torch.int] else nn.MSELoss(reduction='none')
        
        y_for_loss = y.squeeze(-1) if y.dim() > 1 else y
        per_sample_loss = loss_fn(predictions, y_for_loss)
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=1)
        
        env_ids = e.long() if e.dim() == 1 else e.argmax(dim=1)
        env_risks = torch.stack([per_sample_loss[env_ids == eid].mean() for eid in env_ids.unique()])
        pred_loss = self.cvar(env_risks)
        
        violations = self.violations(phi_x, y, e)
        v_tensor = torch.stack([violations.v1, violations.v2, violations.v3])
        softmin_val, weights = self.adaptive_weights(v_tensor)
        
        total_loss = pred_loss + self.lambda_inv * softmin_val
        
        if return_components:
            return {'total': total_loss, 'pred_loss': pred_loss, 'inv_penalty': softmin_val,
                    'v1': violations.v1, 'v2': violations.v2, 'v3': violations.v3,
                    'alpha1': weights[0], 'alpha2': weights[1], 'alpha3': weights[2], 'beta': self.beta}
        return total_loss
    
    def compute_violations_only(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> StructureViolation:
        return self.violations(phi_x, y, e)
    
    def identify_structure(self, phi_x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        violations = self.violations(phi_x, y, e)
        v_tensor = torch.stack([violations.v1, violations.v2, violations.v3])
        return v_tensor.argmin().item() + 1, {'v1': violations.v1.item(), 'v2': violations.v2.item(), 'v3': violations.v3.item()}
    
    def get_structure_weights(self) -> torch.Tensor:
        return self.adaptive_weights.weight_history[-1] if self.adaptive_weights.weight_history.shape[0] > 0 else torch.ones(3) / 3
    
    def get_predicted_structure(self) -> int:
        return self.adaptive_weights.get_predicted_structure()
    
    def get_concentration(self) -> float:
        return self.adaptive_weights.get_concentration()
