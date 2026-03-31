"""Synthetic data generation for the three canonical causal structures
and misspecified bidirectional feedback DGP."""
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
                             noise_scales: Optional[List[float]] = None,
                             label_priors: Optional[List[float]] = None,
                             latent_dim: int = 5, seed: Optional[int] = None) -> SyntheticData:
    """G1: Y -> X <- E. Correct invariance: phi(X) perp E | Y

    Paper: X = W1 * onehot(Y) + W2 * Z + eps, where Z ~ N(0, I_5),
    eps ~ N(0, sigma_e^2 I_d), sigma_e varies across environments.
    Label priors P(Y=1|E=e) vary across environments (label shift),
    ensuring Y is NOT independent of E.
    """
    if seed is not None:
        torch.manual_seed(seed)
    noise_scales = noise_scales or [0.5, 1.0, 1.5][:n_envs]
    label_priors = label_priors or [0.3, 0.5, 0.7][:n_envs]
    samples_per_env = n_samples // n_envs
    W_yx = torch.randn(2, n_features) * 0.5
    W_zx = torch.randn(latent_dim, n_features) * 0.5  # latent Z -> X
    X_list, Y_list, E_list = [], [], []

    for env_id, sigma_e in enumerate(noise_scales):
        p_y1 = label_priors[env_id] if env_id < len(label_priors) else 0.5
        Y_env = torch.bernoulli(torch.full((samples_per_env,), p_y1)).long()
        Y_oh = torch.nn.functional.one_hot(Y_env, 2).float()
        Z_env = torch.randn(samples_per_env, latent_dim)  # invariant latent
        X_env = Y_oh @ W_yx + Z_env @ W_zx + torch.randn(samples_per_env, n_features) * sigma_e
        X_list.append(X_env)
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))

    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list),
                         structure=1, latents={'W_yx': W_yx, 'W_zx': W_zx})


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
    w_xperp_y = torch.randn(d_perp) * 0.8
    W_u_xperp = torch.randn(confounder_dim, d_perp) * 0.5
    X_list, Y_list, E_list, U_list = [], [], [], []

    env_shifts = [torch.zeros(confounder_dim), torch.tensor([1.5] + [0.0]*(confounder_dim-1)),
                  torch.tensor([-1.5] + [0.0]*(confounder_dim-1))][:n_envs]
    for env_id, sigma_e in enumerate(noise_scales):
        U_env = torch.randn(samples_per_env, confounder_dim) + env_shifts[env_id]
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


def _make_bidirectional_weights(n_features, confounder_dim):
    """Create fixed weight matrices for bidirectional DGP (deterministic, seed=42)."""
    g = torch.Generator().manual_seed(42)
    W1 = torch.randn(n_features, confounder_dim, generator=g) * 0.5
    W2 = torch.randn(n_features, 2, generator=g) * 0.5
    w_xy = torch.randn(n_features, generator=g) * 0.5
    w_uy = torch.randn(confounder_dim, generator=g) * 0.5
    w_spur = torch.randn(n_features, generator=g) * 0.3
    return W1, W2, w_xy, w_uy, w_spur


def _generate_bidirectional_env(n, n_features, confounder_dim, alpha, mu_e, sigma_e,
                                 W1, W2, w_xy, w_uy, w_spur=None, spur_strength=0.0):
    """Generate one environment of bidirectional data using pure torch.

    Includes environment-dependent spurious features (spur_strength * Y * w_spur)
    that create a correlation between Y and X that varies across environments.
    Higher label noise (1.5) makes the causal signal weaker relative to spurious features.
    """
    U = torch.randn(n, confounder_dim) + mu_e
    eps_x = torch.randn(n, n_features) * sigma_e * 0.3
    X_init = U @ W1.T + eps_x

    logits_y = X_init @ w_xy + U @ w_uy + torch.randn(n) * 1.5
    Y_prob = torch.sigmoid(logits_y)
    Y_env = torch.bernoulli(Y_prob).long()

    Y_onehot = torch.nn.functional.one_hot(Y_env, 2).float()
    X_feedback = Y_onehot @ W2.T
    X_env = (1 - alpha) * X_init + alpha * X_feedback + eps_x * 0.1

    # Add environment-dependent spurious features
    if w_spur is not None and spur_strength != 0:
        X_env = X_env + spur_strength * (2 * Y_env.float() - 1).unsqueeze(1) * w_spur.unsqueeze(0)

    return X_env.float(), Y_env, U


def generate_bidirectional_data(n_samples: int, n_features: int = 10, n_envs: int = 3,
                                confounder_dim: int = 3, alpha: float = 0.5,
                                seed: Optional[int] = None) -> SyntheticData:
    """Bidirectional feedback DGP: X <-> Y (violates canonical structure assumption).

    Stage 1: Generate initial X from confounder U, then Y from X and U
    Stage 2: Y feeds back into X (creating bidirectional effect)

    Includes environment-dependent spurious features whose correlation with Y
    varies across training environments (strong→weak), creating a distribution
    shift that ERM fails to handle OOD.

    alpha=0 reduces to pure G2 (X->Y), alpha=0.5 is genuinely bidirectional,
    alpha=1 reduces to pure G1 (Y->X).
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu_e_options = [torch.tensor([-1.0, 0, 0]), torch.tensor([0.0, 0, 0]), torch.tensor([1.0, 0, 0])]
    sigma_e_options = [0.5, 1.0, 1.5]
    spur_strengths = [1.5, 1.0, 0.5]  # decreasing spurious strength across envs
    samples_per_env = n_samples // n_envs
    W1, W2, w_xy, w_uy, w_spur = _make_bidirectional_weights(n_features, confounder_dim)

    X_list, Y_list, E_list = [], [], []
    for env_id in range(min(n_envs, len(mu_e_options))):
        X_env, Y_env, _ = _generate_bidirectional_env(
            samples_per_env, n_features, confounder_dim, alpha,
            mu_e_options[env_id], sigma_e_options[env_id],
            W1, W2, w_xy, w_uy, w_spur, spur_strengths[env_id])
        X_list.append(X_env)
        Y_list.append(Y_env)
        E_list.append(torch.full((samples_per_env,), env_id))

    return SyntheticData(X=torch.cat(X_list), Y=torch.cat(Y_list), E=torch.cat(E_list),
                         structure=0, latents={'alpha': alpha})


def generate_bidirectional_test_data(n_samples: int, n_features: int = 10,
                                     confounder_dim: int = 3, alpha: float = 0.5,
                                     shift_magnitude: float = 2.0,
                                     seed: Optional[int] = None) -> SyntheticData:
    """Test environment with larger distribution shift for bidirectional DGP.

    The spurious correlation is reversed (negative strength) in the test env,
    creating distribution shift that penalizes models relying on spurious features.
    """
    if seed is not None:
        torch.manual_seed(seed)

    W1, W2, w_xy, w_uy, w_spur = _make_bidirectional_weights(n_features, confounder_dim)
    mu_test = torch.zeros(confounder_dim)  # no covariate shift (keeps labels balanced)
    sigma_test = 1.0

    X, Y, _ = _generate_bidirectional_env(
        n_samples, n_features, confounder_dim, alpha,
        mu_test, sigma_test, W1, W2, w_xy, w_uy,
        w_spur, spur_strength=-0.8)  # reversed spurious correlation

    return SyntheticData(X=X, Y=Y, E=torch.zeros(n_samples).long(),
                         structure=0, latents={'alpha': alpha, 'shift': shift_magnitude})


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
        seed = self.seed + 1000 if self.seed is not None else None
        if self.structure == 1:
            test_data = generate_anticausal_data(n_test, self.n_features, 1, noise_scales=[2.0 * test_shift], seed=seed)
        elif self.structure == 2:
            test_data = generate_confounded_descendant_data(n_test, self.n_features, 1, env_shifts=[torch.tensor([2.0*test_shift, 0., 0.])], seed=seed)
        else:
            test_data = generate_confounded_outcome_data(n_test, self.n_features, 1, noise_scales=[2.0 * test_shift], seed=seed)
        return train_data, test_data

    def generate_train_val_test(self, n_train: int, n_val: int, n_test: int,
                                val_shift: float = 0.5, test_shift: float = 1.0
                                ) -> Tuple[SyntheticData, SyntheticData, SyntheticData]:
        """Generate train (3 envs), validation (1 env, mild shift), test (1 env, large shift)."""
        train_data = self.generate(n_train)
        seed_val = self.seed + 500 if self.seed is not None else None
        seed_test = self.seed + 1000 if self.seed is not None else None
        if self.structure == 1:
            val_data = generate_anticausal_data(n_val, self.n_features, 1, noise_scales=[1.0 + val_shift],
                                                label_priors=[0.5], seed=seed_val)
            test_data = generate_anticausal_data(n_test, self.n_features, 1, noise_scales=[2.0 * test_shift],
                                                 label_priors=[0.5], seed=seed_test)
        elif self.structure == 2:
            val_data = generate_confounded_descendant_data(
                n_val, self.n_features, 1,
                env_shifts=[torch.tensor([val_shift, 0., 0.])], seed=seed_val)
            test_data = generate_confounded_descendant_data(
                n_test, self.n_features, 1,
                env_shifts=[torch.tensor([2.0 * test_shift, 0., 0.])], seed=seed_test)
        else:
            val_data = generate_confounded_outcome_data(n_val, self.n_features, 1, noise_scales=[1.0 + val_shift], seed=seed_val)
            test_data = generate_confounded_outcome_data(n_test, self.n_features, 1, noise_scales=[2.0 * test_shift], seed=seed_test)
        return train_data, val_data, test_data
