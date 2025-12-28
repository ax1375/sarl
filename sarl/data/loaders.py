"""Data loader utilities."""
import torch
from torch.utils.data import DataLoader, Sampler
from typing import Dict, Iterator, List, Tuple


class BalancedEnvSampler(Sampler):
    """Balanced sampling across environments.

    Args:
        dataset: MultiEnvDataset with env_indices attribute
        batch_size: Total batch size (must be divisible by number of environments)
        drop_last: Whether to drop incomplete batches

    Raises:
        ValueError: If batch_size is not divisible by n_envs or dataset is invalid
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        if not hasattr(dataset, 'env_indices'):
            raise ValueError("Dataset must have 'env_indices' attribute")
        if not hasattr(dataset, 'n_envs'):
            raise ValueError("Dataset must have 'n_envs' attribute")
        if len(dataset.env_indices) == 0:
            raise ValueError("Dataset has no environments")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset, self.batch_size = dataset, batch_size
        self.n_envs = dataset.n_envs

        if batch_size % self.n_envs != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by n_envs ({self.n_envs})")

        self.per_env = batch_size // self.n_envs

        env_sizes = [len(idx) for idx in dataset.env_indices.values()]
        if len(env_sizes) == 0:
            raise ValueError("Dataset has no environment indices")
        if min(env_sizes) < self.per_env:
            raise ValueError(f"Smallest environment has {min(env_sizes)} samples, but need at least {self.per_env}")

        self.n_batches = min(env_sizes) // self.per_env
    
    def __iter__(self) -> Iterator[List[int]]:
        shuffled = {e: idx[torch.randperm(len(idx))] for e, idx in self.dataset.env_indices.items()}
        for batch_idx in range(self.n_batches):
            start, end = batch_idx * self.per_env, (batch_idx + 1) * self.per_env
            yield [i for e in range(self.n_envs) for i in shuffled[e][start:end].tolist()]
    
    def __len__(self) -> int:
        return self.n_batches


def create_multi_env_loaders(train_dataset, test_dataset, batch_size: int = 128, balanced: bool = True,
                             num_workers: int = 0, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create train and test loaders."""
    if balanced:
        train_loader = DataLoader(train_dataset, batch_sampler=BalancedEnvSampler(train_dataset, batch_size),
                                 num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
