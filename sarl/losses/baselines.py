"""Baseline penalty methods for domain generalization.

Implements three standard baselines:
  - IRM (Invariant Risk Minimization, Arjovsky et al. 2019)
  - VREx (Variance Risk Extrapolation, Krueger et al. 2021)
  - CIRCE (Conditional Independence Regression Covariance Estimate)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from ..kernels.base import GaussianKernel, DeltaKernel, compute_median_bandwidth, EPSILON


class IRMPenalty(nn.Module):
    """Invariant Risk Minimization (IRMv1) penalty.

    Penalty per environment e:
        ||grad_{w|w=1.0} R_e(w * predictions)||^2

    Total loss = mean(R_e) + lambda_irm * mean(penalty_e)

    Reference: Arjovsky et al., "Invariant Risk Minimization", 2019.
    """

    def __init__(self, lambda_irm: float = 1.0):
        super().__init__()
        self.lambda_irm = lambda_irm

    def _per_env_loss(self, predictions: torch.Tensor, y: torch.Tensor,
                      loss_fn: nn.Module) -> torch.Tensor:
        """Compute mean loss for a single environment."""
        y_squeezed = y.squeeze(-1) if y.dim() > 1 else y
        per_sample = loss_fn(predictions, y_squeezed)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        return per_sample.mean()

    def _irm_penalty(self, predictions: torch.Tensor, y: torch.Tensor,
                     loss_fn: nn.Module) -> torch.Tensor:
        """Compute IRMv1 gradient penalty for one environment.

        Scales predictions by a dummy scalar w=1.0 and computes
        ||grad_w R_e(w * predictions)||^2.
        """
        w = torch.tensor(1.0, device=predictions.device, requires_grad=True)
        scaled = predictions * w
        risk = self._per_env_loss(scaled, y, loss_fn)
        grad = torch.autograd.grad(risk, w, create_graph=True)[0]
        return grad.pow(2)

    def compute_penalty_only(self, predictions: torch.Tensor, y: torch.Tensor,
                             e: torch.Tensor, phi_x: torch.Tensor = None,
                             loss_fn: nn.Module = None) -> torch.Tensor:
        """Return the IRM penalty term (without the ERM loss)."""
        if loss_fn is None:
            loss_fn = self._default_loss(y)
        env_ids = e.long() if e.dim() == 1 else e.argmax(dim=1)
        penalties = []
        for eid in env_ids.unique():
            mask = env_ids == eid
            penalties.append(self._irm_penalty(predictions[mask], y[mask], loss_fn))
        return torch.stack(penalties).mean()

    def forward(self, predictions: torch.Tensor, y: torch.Tensor,
                e: torch.Tensor, phi_x: torch.Tensor = None,
                loss_fn: nn.Module = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_fn is None:
            loss_fn = self._default_loss(y)
        env_ids = e.long() if e.dim() == 1 else e.argmax(dim=1)

        env_risks = []
        env_penalties = []
        for eid in env_ids.unique():
            mask = env_ids == eid
            env_risks.append(self._per_env_loss(predictions[mask], y[mask], loss_fn))
            env_penalties.append(self._irm_penalty(predictions[mask], y[mask], loss_fn))

        mean_risk = torch.stack(env_risks).mean()
        mean_penalty = torch.stack(env_penalties).mean()
        total = mean_risk + self.lambda_irm * mean_penalty

        if return_components:
            return {'total': total, 'erm_loss': mean_risk, 'penalty': mean_penalty}
        return total

    @staticmethod
    def _default_loss(y: torch.Tensor) -> nn.Module:
        if y.dtype in (torch.long, torch.int):
            return nn.CrossEntropyLoss(reduction='none')
        return nn.MSELoss(reduction='none')


class VRExPenalty(nn.Module):
    """Variance Risk Extrapolation (VREx) penalty.

    Penalty = Var({R_e}) across environments.
    Total loss = mean(R_e) + lambda_vrex * Var(R_e)

    Reference: Krueger et al., "Out-of-Distribution Generalization via
    Risk Extrapolation", 2021.
    """

    def __init__(self, lambda_vrex: float = 1.0):
        super().__init__()
        self.lambda_vrex = lambda_vrex

    def compute_penalty_only(self, predictions: torch.Tensor, y: torch.Tensor,
                             e: torch.Tensor, phi_x: torch.Tensor = None,
                             loss_fn: nn.Module = None) -> torch.Tensor:
        """Return Var(R_e) without the ERM loss."""
        if loss_fn is None:
            loss_fn = self._default_loss(y)
        env_risks = self._per_env_risks(predictions, y, e, loss_fn)
        return env_risks.var()

    def forward(self, predictions: torch.Tensor, y: torch.Tensor,
                e: torch.Tensor, phi_x: torch.Tensor = None,
                loss_fn: nn.Module = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_fn is None:
            loss_fn = self._default_loss(y)
        env_risks = self._per_env_risks(predictions, y, e, loss_fn)

        mean_risk = env_risks.mean()
        variance = env_risks.var()
        total = mean_risk + self.lambda_vrex * variance

        if return_components:
            return {'total': total, 'erm_loss': mean_risk, 'penalty': variance}
        return total

    def _per_env_risks(self, predictions: torch.Tensor, y: torch.Tensor,
                       e: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        y_squeezed = y.squeeze(-1) if y.dim() > 1 else y
        per_sample = loss_fn(predictions, y_squeezed)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        env_ids = e.long() if e.dim() == 1 else e.argmax(dim=1)
        return torch.stack([per_sample[env_ids == eid].mean() for eid in env_ids.unique()])

    @staticmethod
    def _default_loss(y: torch.Tensor) -> nn.Module:
        if y.dtype in (torch.long, torch.int):
            return nn.CrossEntropyLoss(reduction='none')
        return nn.MSELoss(reduction='none')


class CIRCEPenalty(nn.Module):
    """Conditional Independence Regression Covariance Estimate (CIRCE).

    Tests phi(X) _|_ E | Y using the Hilbert-Schmidt norm of the
    conditional cross-covariance operator C_{phi,E|Y}.

    Steps:
      1. Compute kernel matrices K_phi for phi(X) and K_E for E.
      2. Residualize both on Y via kernel ridge regression.
      3. Penalty = ||C_{phi,E|Y}||^2_HS = (1/n^2) * ||K_phi_res @ K_E_res||_F^2

    This is analogous to V1 in SaCRL's violation metrics but used as a
    standalone fixed penalty.

    Total loss = mean(R_e) + lambda_circe * CIRCE_penalty
    """

    def __init__(self, lambda_circe: float = 1.0, ridge_lambda: float = 1e-3):
        super().__init__()
        self.lambda_circe = lambda_circe
        self.ridge_lambda = ridge_lambda
        self.kernel_phi = GaussianKernel()
        self.kernel_e = DeltaKernel()
        self.kernel_y = GaussianKernel()

    def _kernel_residual(self, K: torch.Tensor, K_Y: torch.Tensor) -> torch.Tensor:
        """Residualize a kernel matrix on Y via kernel ridge regression.

        Given kernel matrix K (n x n) for some variable and kernel matrix
        K_Y (n x n) for Y, compute residual kernel:
            K_res = K - K_Y (K_Y + gamma I)^{-1} K

        This removes the component of the variable explained by Y.
        """
        n = K.shape[0]
        d = 1  # Y is typically low-dimensional
        gamma = n ** (-1.0 / (2 + d))
        K_Y_reg = K_Y + gamma * torch.eye(n, device=K.device)
        try:
            alpha = torch.linalg.solve(K_Y_reg, K)
        except (RuntimeError, torch.linalg.LinAlgError):
            alpha = torch.linalg.lstsq(K_Y_reg, K).solution
        return K - K_Y @ alpha

    def _compute_circe(self, phi_x: torch.Tensor, y: torch.Tensor,
                       e: torch.Tensor) -> torch.Tensor:
        """Compute the CIRCE penalty: ||C_{phi,E|Y}||^2_HS."""
        n = phi_x.shape[0]
        if y.dim() == 1:
            y = y.unsqueeze(1).float()
        else:
            y = y.float()

        # Handle environment encoding
        if e.dim() == 1:
            num_envs = int(e.max().item()) + 1
            e_onehot = F.one_hot(e.long(), num_envs).float()
        else:
            e_onehot = e.float()

        # Compute kernel matrices
        K_phi = self.kernel_phi(phi_x)
        K_E = self.kernel_e(e_onehot)
        K_Y = self.kernel_y(y)

        # Residualize on Y
        K_phi_res = self._kernel_residual(K_phi, K_Y)
        K_E_res = self._kernel_residual(K_E, K_Y)

        # Center the residual kernels
        H = torch.eye(n, device=K_phi.device) - torch.ones(n, n, device=K_phi.device) / n
        K_phi_c = H @ K_phi_res @ H
        K_E_c = H @ K_E_res @ H

        # HS norm of cross-covariance: (1/n^2) * trace(K_phi_c @ K_E_c @ K_phi_c @ K_E_c)
        # Simplified: (1/n^2) * ||K_phi_c @ K_E_c||_F^2
        product = K_phi_c @ K_E_c
        penalty = (product * product).sum() / (n * n)
        return torch.clamp(penalty, min=0.0)

    def compute_penalty_only(self, predictions: torch.Tensor, y: torch.Tensor,
                             e: torch.Tensor, phi_x: torch.Tensor,
                             loss_fn: nn.Module = None) -> torch.Tensor:
        """Return the CIRCE penalty without the ERM loss."""
        return self._compute_circe(phi_x, y, e)

    def forward(self, predictions: torch.Tensor, y: torch.Tensor,
                e: torch.Tensor, phi_x: torch.Tensor,
                loss_fn: nn.Module = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_fn is None:
            loss_fn = self._default_loss(y)

        # ERM component: mean of per-environment risks
        y_squeezed = y.squeeze(-1) if y.dim() > 1 else y
        per_sample = loss_fn(predictions, y_squeezed)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        env_ids = e.long() if e.dim() == 1 else e.argmax(dim=1)
        env_risks = torch.stack([per_sample[env_ids == eid].mean() for eid in env_ids.unique()])
        mean_risk = env_risks.mean()

        # CIRCE penalty
        penalty = self._compute_circe(phi_x, y, e)
        total = mean_risk + self.lambda_circe * penalty

        if return_components:
            return {'total': total, 'erm_loss': mean_risk, 'penalty': penalty}
        return total

    @staticmethod
    def _default_loss(y: torch.Tensor) -> nn.Module:
        if y.dtype in (torch.long, torch.int):
            return nn.CrossEntropyLoss(reduction='none')
        return nn.MSELoss(reduction='none')


def train_baseline(model: nn.Module, train_loader, method: nn.Module,
                   n_epochs: int = 100, lambda_pen: float = 1.0,
                   lr: float = 1e-3, device: str = 'cpu') -> Dict[str, list]:
    """Train a model with a baseline penalty method.

    Args:
        model: Neural network with a ``featurizer`` attribute that returns
            representations phi(X) and a forward pass returning predictions.
            If the model does not have a ``featurizer`` attribute, phi_x is
            set to the predictions themselves (sufficient for IRM and VREx).
        train_loader: DataLoader yielding (x, y, e) batches where e is the
            environment index.
        method: One of IRMPenalty, VRExPenalty, or CIRCEPenalty (already
            constructed with the desired lambda).
        n_epochs: Number of training epochs.
        lambda_pen: Penalty weight (overrides method.lambda_irm /
            method.lambda_vrex / method.lambda_circe for convenience).
        lr: Learning rate for Adam optimizer.
        device: Device string ('cpu', 'cuda', etc.).

    Returns:
        Dictionary with lists of per-epoch metrics:
            'total_loss', 'erm_loss', 'penalty'.
    """
    # Override lambda on the method object for convenience
    if isinstance(method, IRMPenalty):
        method.lambda_irm = lambda_pen
    elif isinstance(method, VRExPenalty):
        method.lambda_vrex = lambda_pen
    elif isinstance(method, CIRCEPenalty):
        method.lambda_circe = lambda_pen

    model = model.to(device)
    method = method.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'total_loss': [], 'erm_loss': [], 'penalty': []}

    for epoch in range(n_epochs):
        epoch_total, epoch_erm, epoch_pen = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            x, y, e = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Forward pass
            if hasattr(model, 'featurizer'):
                phi_x = model.featurizer(x)
                predictions = model(x)
            else:
                predictions = model(x)
                phi_x = predictions

            result = method(predictions=predictions, y=y, e=e, phi_x=phi_x,
                            return_components=True)

            loss = result['total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += result['total'].item()
            epoch_erm += result['erm_loss'].item()
            epoch_pen += result['penalty'].item()
            n_batches += 1

        if n_batches > 0:
            history['total_loss'].append(epoch_total / n_batches)
            history['erm_loss'].append(epoch_erm / n_batches)
            history['penalty'].append(epoch_pen / n_batches)

    return history
