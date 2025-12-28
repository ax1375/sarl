"""Dataset classes for multi-environment learning."""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class MultiEnvDataset(Dataset):
    """Multi-environment dataset wrapper."""
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, E: torch.Tensor):
        self.X, self.Y, self.E = X.float(), Y, E.long()
        assert len(X) == len(Y) == len(E)
        self.n_envs = int(E.max().item()) + 1
        self.env_indices = {int(e): (E == e).nonzero(as_tuple=True)[0] for e in range(self.n_envs)}
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx], self.E[idx]


class ColoredMNIST(Dataset):
    """Colored MNIST for spurious correlation experiments."""
    def __init__(self, root: str = './data', env: str = 'train1', correlation: Optional[float] = None):
        self.correlation = correlation or {'train1': 0.9, 'train2': 0.8, 'test': 0.1}.get(env, 0.5)
        self.images, self.labels = self._load_mnist(root, env)
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
        n = len(self.labels)
        flip = torch.bernoulli(torch.ones(n) * (1 - self.correlation)).bool()
        colors = self.labels.clone()
        colors[flip] = 1 - colors[flip]
        colored = torch.zeros(n, 3, 28, 28)
        for i in range(n):
            colored[i, colors[i].item()] = self.images[i]
        return colored
    
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
        indices = torch.cat([idx[torch.randperm(len(idx))[:subsample]] for e, idx in 
                           {e: (E_train == e).nonzero(as_tuple=True)[0] for e in range(len(train_correlations))}.items()])
        X_train, Y_train, E_train = X_train[indices], Y_train[indices], E_train[indices]
        perm = torch.randperm(len(X_test))[:subsample]
        X_test, Y_test, E_test = X_test[perm], Y_test[perm], E_test[perm]
    
    return MultiEnvDataset(X_train, Y_train, E_train), MultiEnvDataset(X_test, Y_test, E_test)
