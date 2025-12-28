"""Synthetic data generation for the three canonical causal structures."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SyntheticData:
    X: torch.Tensor
    Y: torch.Tensor
    E: torch.Tensor
    structure: int
    latents: Optional[Dict[str, torch.Tensor]] = None


def generate_anticausal_data(n_samples: int, n_features: int = 10, n_envs: int = 3,
                             noise_scales: Optional[List[float]] = None, seed: Optional[int] = None) -> SyntheticData:
    """G1: Y → X ← E. Correct invariance: φ(X) ⊥ E | Y"""
    if seed is not None:
        torch.manual_seed(seed)
    noise_scales = noise_scales or [0.5, 1.0, 1.5][:n_envs]
    samples_per_env = n_samples // n_envs
    W_yx = torch.randn(2, n_features) * 0.5
    X_list, Y_list, E_list = [], [], []
    
    for env_id, sigma_e in enumerate(noise_scales):
        Y_env = torch.randint(0, 2, (samples_per_env,))
        Y_oh = torch.nn.functional.one_hot(Y_env, 2).float()
        X_env = Y_oh @ W_yx + torch.randn(samples_per_env, n_features) * sigma_e
        X_list.append(X_env)
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))
    
    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list), structure=1, latents={'W_yx': W_yx})


def generate_confounded_descendant_data(n_samples: int, n_features: int = 10, n_envs: int = 3,
                                        confounder_dim: int = 3, env_shifts: Optional[List[torch.Tensor]] = None,
                                        seed: Optional[int] = None) -> SyntheticData:
    """G2: U → X, U → Y, X → Y. Correct invariance: Y ⊥ E | φ(X)"""
    if seed is not None:
        torch.manual_seed(seed)
    env_shifts = env_shifts or [torch.zeros(confounder_dim), torch.tensor([1.0] + [0.0]*(confounder_dim-1)), 
                                torch.tensor([-1.0] + [0.0]*(confounder_dim-1))][:n_envs]
    samples_per_env = n_samples // n_envs
    W_ux = torch.randn(confounder_dim, n_features) * 0.5
    w_xy = torch.randn(n_features) * 0.3
    w_uy = torch.randn(confounder_dim) * 0.5
    X_list, Y_list, E_list, U_list = [], [], [], []
    
    for env_id, mu_u in enumerate(env_shifts):
        U_env = torch.randn(samples_per_env, confounder_dim) + mu_u
        X_env = U_env @ W_ux + torch.randn(samples_per_env, n_features) * 0.1
        logits = X_env @ w_xy + U_env @ w_uy
        Y_env = torch.bernoulli(torch.sigmoid(logits)).long()
        X_list.append(X_env)
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))
        U_list.append(U_env)
    
    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list), structure=2,
                         latents={'U': torch.cat(U_list), 'W_ux': W_ux, 'w_xy': w_xy, 'w_uy': w_uy})


def generate_confounded_outcome_data(n_samples: int, n_features: int = 10, n_envs: int = 3,
                                     confounder_dim: int = 3, noise_scales: Optional[List[float]] = None,
                                     seed: Optional[int] = None) -> SyntheticData:
    """G3: U → Y, U → Z → X_z, X_z^⊥ → Y. Correct invariance: φ(X) ⊥ E"""
    if seed is not None:
        torch.manual_seed(seed)
    noise_scales = noise_scales or [0.5, 1.0, 1.5][:n_envs]
    samples_per_env = n_samples // n_envs
    d_z, d_perp = n_features // 2, n_features - n_features // 2
    W_uz = torch.randn(confounder_dim, d_z) * 0.5
    W_yxz = torch.randn(2, d_z) * 0.3
    w_xperp_y = torch.randn(d_perp) * 0.5
    W_u_xperp = torch.randn(confounder_dim, d_perp) * 0.3
    X_list, Y_list, E_list, U_list = [], [], [], []
    
    for env_id, sigma_e in enumerate(noise_scales):
        U_env = torch.randn(samples_per_env, confounder_dim)
        X_perp = U_env @ W_u_xperp + torch.randn(samples_per_env, d_perp) * sigma_e
        Y_env = torch.bernoulli(torch.sigmoid(X_perp @ w_xperp_y)).long()
        Y_oh = torch.nn.functional.one_hot(Y_env, 2).float()
        X_z = Y_oh @ W_yxz + U_env @ W_uz + torch.randn(samples_per_env, d_z) * 0.1
        X_list.append(torch.cat([X_z, X_perp], dim=1))
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))
        U_list.append(U_env)
    
    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list), structure=3,
                         latents={'U': torch.cat(U_list), 'd_z': d_z, 'd_perp': d_perp})


class SyntheticDataGenerator:
    """Unified interface for synthetic data generation."""
    def __init__(self, structure: Union[int, str] = 1, n_features: int = 10, n_envs: int = 3, seed: Optional[int] = None):
        self.structure, self.n_features, self.n_envs, self.seed = structure, n_features, n_envs, seed
    
    def generate(self, n_samples: int) -> SyntheticData:
        """Generate synthetic data for the specified causal structure.

        Args:
            n_samples: Number of samples to generate

        Returns:
            SyntheticData with generated X, Y, E and structure information

        Raises:
            ValueError: If structure is not in {1, 2, 3}
        """
        if self.structure == 1:
            return generate_anticausal_data(n_samples, self.n_features, self.n_envs, seed=self.seed)
        elif self.structure == 2:
            return generate_confounded_descendant_data(n_samples, self.n_features, self.n_envs, seed=self.seed)
        elif self.structure == 3:
            return generate_confounded_outcome_data(n_samples, self.n_features, self.n_envs, seed=self.seed)
        else:
            raise ValueError(f"Unknown structure: {self.structure}. Must be 1, 2, or 3.")
    
    def generate_train_test(self, n_train: int, n_test: int, test_shift: float = 1.0) -> Tuple[SyntheticData, SyntheticData]:
        train_data = self.generate(n_train)
        seed = self.seed + 1000 if self.seed else None
        if self.structure == 1:
            test_data = generate_anticausal_data(n_test, self.n_features, 1, [2.0 * test_shift], seed)
        elif self.structure == 2:
            test_data = generate_confounded_descendant_data(n_test, self.n_features, 1, env_shifts=[torch.tensor([2.0*test_shift, 0., 0.])], seed=seed)
        else:
            test_data = generate_confounded_outcome_data(n_test, self.n_features, 1, [2.0 * test_shift], seed)
        return train_data, test_data
