"""SoftMin aggregation for differentiable structure selection."""
import torch
import torch.nn as nn
from typing import List, Tuple, Union


class SoftMin(nn.Module):
    """Differentiable SoftMin: SoftMin_β({V_k}) = -1/β log Σ exp(-β V_k)"""
    def __init__(self, beta: float = 10.0, learnable: bool = False, beta_min: float = 0.1, beta_max: float = 100.0):
        super().__init__()
        self.beta_min, self.beta_max = beta_min, beta_max
        if learnable:
            self.log_beta = nn.Parameter(torch.tensor(float(beta)).log())
        else:
            self.register_buffer('log_beta', torch.tensor(float(beta)).log())
    
    @property
    def beta(self) -> torch.Tensor:
        return torch.clamp(self.log_beta.exp(), self.beta_min, self.beta_max)
    
    def set_beta(self, beta: float):
        self.log_beta.data = torch.tensor(float(beta)).log().to(self.log_beta.device)
    
    def forward(self, violations: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(violations, (list, tuple)):
            violations = torch.stack([v if torch.is_tensor(v) else torch.tensor(v) for v in violations])
        violations = violations.float()
        v_max = violations.max()
        v_shifted = violations - v_max
        log_sum_exp = torch.logsumexp(-self.beta * v_shifted, dim=0)
        return v_max - log_sum_exp / self.beta
    
    def get_weights(self, violations: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(violations, (list, tuple)):
            violations = torch.stack([v if torch.is_tensor(v) else torch.tensor(v) for v in violations])
        return torch.softmax(-self.beta * violations.float(), dim=0)


class AdaptiveWeights(nn.Module):
    """Compute and track adaptive structure weights."""
    def __init__(self, beta: float = 10.0, num_structures: int = 3):
        super().__init__()
        self.softmin = SoftMin(beta=beta)
        self.num_structures = num_structures
        self.register_buffer('weight_history', torch.zeros(0, num_structures))
        self.register_buffer('violation_history', torch.zeros(0, num_structures))
    
    @property
    def beta(self) -> torch.Tensor:
        return self.softmin.beta
    
    def set_beta(self, beta: float):
        self.softmin.set_beta(beta)
    
    def forward(self, violations: Union[torch.Tensor, List[torch.Tensor]], track: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(violations, (list, tuple)):
            violations = torch.stack([v if torch.is_tensor(v) else torch.tensor(v) for v in violations])
        softmin_val = self.softmin(violations)
        weights = self.softmin.get_weights(violations)
        if track and self.training:
            self.violation_history = torch.cat([self.violation_history, violations.detach().unsqueeze(0)], dim=0)[-1000:]
            self.weight_history = torch.cat([self.weight_history, weights.detach().unsqueeze(0)], dim=0)[-1000:]
        return softmin_val, weights
    
    def get_predicted_structure(self) -> int:
        if self.weight_history.shape[0] == 0:
            return 1
        return self.weight_history[-100:].mean(dim=0).argmax().item() + 1
    
    def get_concentration(self) -> float:
        if self.weight_history.shape[0] == 0:
            return 0.0
        max_weight = self.weight_history[-100:].mean(dim=0).max().item()
        return max(0.0, (max_weight - 1/self.num_structures) / (1 - 1/self.num_structures))


class TemperatureScheduler:
    """Schedule temperature β during training."""
    def __init__(self, beta_start: float = 1.0, beta_end: float = 50.0, warmup_epochs: int = 10, total_epochs: int = 100, schedule: str = 'linear'):
        self.beta_start, self.beta_end = beta_start, beta_end
        self.warmup_epochs, self.total_epochs = warmup_epochs, total_epochs
        self.schedule = schedule
    
    def get_beta(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.beta_start
        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs))
        if self.schedule == 'linear':
            return self.beta_start + progress * (self.beta_end - self.beta_start)
        elif self.schedule == 'exponential':
            return (torch.tensor(self.beta_start).log() + progress * (torch.tensor(self.beta_end).log() - torch.tensor(self.beta_start).log())).exp().item()
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def step(self, softmin_module: SoftMin, epoch: int) -> float:
        beta = self.get_beta(epoch)
        softmin_module.set_beta(beta)
        return beta
