"""Dataset classes for multi-environment learning."""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class MultiEnvDataset(Dataset):
    """Multi-environment dataset wrapper.

    Args:
        X: Input features, shape (n, d)
        Y: Labels, shape (n,)
        E: Environment indicators, shape (n,)

    Raises:
        ValueError: If inputs are empty or have mismatched sizes
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, E: torch.Tensor):
        if X.numel() == 0 or Y.numel() == 0 or E.numel() == 0:
            raise ValueError("Input tensors cannot be empty")

        if not (len(X) == len(Y) == len(E)):
            raise ValueError(f"Input tensors must have same length. Got X: {len(X)}, Y: {len(Y)}, E: {len(E)}")

        self.X, self.Y, self.E = X.float(), Y, E.long()
        self.n_envs = int(E.max().item()) + 1
        self.env_indices = {int(e): (E == e).nonzero(as_tuple=True)[0] for e in range(self.n_envs)}
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx], self.E[idx]


class ColoredMNIST(Dataset):
    """Colored MNIST following Arjovsky et al. (2019) exactly.

    Protocol:
      1. Preliminary label Y* = (digit >= 5)
      2. Final label Y = Y* XOR Bernoulli(label_noise)  -- 25% noise
      3. Color = Y XOR Bernoulli(1 - correlation)
      4. Image = 2-channel (digit in one channel, zero in other)

    The digit is placed in channel 0 (red) or channel 1 (green) based on
    the color assignment. The other channel is all zeros. This makes color
    trivially easy to learn (just check which channel has content), while
    shape requires analyzing pixel patterns within the active channel.

    With label_noise=0.25:
      - Shape accuracy ceiling = 75%
      - Color accuracy = correlation (90%/80% train, 10% test)
      - ERM learns color → ~10% OOD
      - IRM/invariant methods learn shape → ~72% OOD
    """
    def __init__(self, root: str = './data', env: str = 'train1',
                 correlation: Optional[float] = None, label_noise: float = 0.25):
        self.correlation = correlation or {'train1': 0.9, 'train2': 0.8, 'test': 0.1}.get(env, 0.5)
        self.label_noise = label_noise
        self.images, self.labels = self._load_mnist(root, env)
        # Apply label noise: flip Y* with probability label_noise
        if self.label_noise > 0:
            flip_label = torch.bernoulli(torch.ones(len(self.labels)) * self.label_noise).bool()
            self.labels[flip_label] = 1 - self.labels[flip_label]
        self.colored_images = self._colorize()

    def _load_mnist(self, root: str, env: str) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from torchvision import datasets, transforms
            mnist = datasets.MNIST(root, train=env != 'test', download=True, transform=transforms.ToTensor())
            return mnist.data.float() / 255.0, (mnist.targets >= 5).long()
        except ImportError:
            n = 10000 if env != 'test' else 2000
            return torch.rand(n, 28, 28), torch.randint(0, 2, (n,))

    def _colorize(self) -> torch.Tensor:
        """Colorize following original IRM paper: digit in one of 2 channels.

        For each image:
          - Both channels start with the same grayscale digit
          - The channel NOT matching the color assignment is zeroed out
          - Result: digit visible in exactly one channel, color = which channel

        Returns:
            2-channel images of shape (n, 2, 28, 28)
        """
        n = len(self.labels)
        # Determine spurious color: flip with probability (1 - correlation)
        flip = torch.bernoulli(torch.ones(n) * (1 - self.correlation)).bool()
        colors = self.labels.clone()
        colors[flip] = 1 - colors[flip]

        # Stack digit into both channels: (n, 2, 28, 28)
        imgs = torch.stack([self.images, self.images], dim=1)
        # Zero out the channel that doesn't match the color
        # colors=0 → keep channel 0, zero channel 1
        # colors=1 → keep channel 1, zero channel 0
        for i in range(n):
            imgs[i, (1 - colors[i]).long(), :, :] *= 0

        return imgs
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.colored_images[idx], self.labels[idx]


def create_colored_mnist(root: str = './data', train_correlations: List[float] = [0.9, 0.8],
                         test_correlation: float = 0.1, subsample: Optional[int] = None) -> Tuple[MultiEnvDataset, MultiEnvDataset]:
    """Create Colored MNIST train/test datasets."""
    train_datasets = [ColoredMNIST(root, f'train{i+1}', corr) for i, corr in enumerate(train_correlations)]
    X_train = torch.cat([ds.colored_images for ds in train_datasets])
    Y_train = torch.cat([ds.labels for ds in train_datasets])
    E_train = torch.cat([torch.full((len(ds),), i) for i, ds in enumerate(train_datasets)])
    
    test_ds = ColoredMNIST(root, 'test', test_correlation)
    X_test, Y_test, E_test = test_ds.colored_images, test_ds.labels, torch.zeros(len(test_ds))
    
    if subsample:
        # Subsample from each environment separately for balanced training
        env_indices = {e: (E_train == e).nonzero(as_tuple=True)[0] for e in range(len(train_correlations))}
        sampled_indices = []
        for e, idx in env_indices.items():
            n_samples = min(subsample, len(idx))
            sampled_indices.append(idx[torch.randperm(len(idx))[:n_samples]])
        indices = torch.cat(sampled_indices)
        X_train, Y_train, E_train = X_train[indices], Y_train[indices], E_train[indices]

        # Subsample test set
        n_test_samples = min(subsample, len(X_test))
        perm = torch.randperm(len(X_test))[:n_test_samples]
        X_test, Y_test, E_test = X_test[perm], Y_test[perm], E_test[perm]
    
    return MultiEnvDataset(X_train, Y_train, E_train), MultiEnvDataset(X_test, Y_test, E_test)
