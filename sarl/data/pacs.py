"""PACS dataset loader with ResNet18 feature extraction for domain generalization."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from typing import Dict, List, Optional, Tuple

from .datasets import MultiEnvDataset

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_CLASS_TO_IDX = {cls: i for i, cls in enumerate(PACS_CLASSES)}


class PACSImageDataset(Dataset):
    """Raw PACS image dataset loading from directory structure.

    Expected structure: data_dir/{domain}/{class_name}/*.jpg
    """
    def __init__(self, data_dir: str, domains: List[str], transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.samples = []  # (path, label, domain_idx)

        for domain_idx, domain in enumerate(domains):
            domain_dir = os.path.join(data_dir, domain)
            if not os.path.isdir(domain_dir):
                raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
            for cls_name in PACS_CLASSES:
                cls_dir = os.path.join(domain_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(cls_dir, fname),
                            PACS_CLASS_TO_IDX[cls_name],
                            domain_idx,
                        ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, domain_idx = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, domain_idx


def _get_resnet18_feature_extractor(device: str = 'cpu') -> nn.Module:
    """Get a pretrained ResNet18 without the final classification layer.

    Returns a model that outputs 512-dimensional feature vectors.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove final FC layer, keep everything up to avgpool
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def _extract_features(dataset: PACSImageDataset, model: nn.Module,
                      device: str = 'cpu', batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from images using the given model.

    Returns:
        X: features (n, 512)
        Y: labels (n,)
        E: environment/domain indices (n,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []
    all_envs = []

    with torch.no_grad():
        for images, labels, domain_indices in loader:
            images = images.to(device)
            features = model(images).cpu()
            all_features.append(features)
            all_labels.append(labels)
            all_envs.append(domain_indices)

    X = torch.cat(all_features, dim=0)
    Y = torch.cat(all_labels, dim=0)
    E = torch.cat(all_envs, dim=0)
    return X, Y, E


def get_pacs_features(data_dir: str, target_domain: str, device: str = 'cpu',
                      force_extract: bool = False, batch_size: int = 64
                      ) -> Tuple[MultiEnvDataset, MultiEnvDataset]:
    """Load PACS features (extract with ResNet18 if not cached).

    Uses leave-one-domain-out: trains on all domains except target_domain,
    tests on target_domain.

    Args:
        data_dir: Path to PACS root directory containing domain subdirectories.
        target_domain: Domain to hold out for testing (e.g., 'sketch').
        device: Device for feature extraction ('cpu' or 'cuda').
        force_extract: If True, re-extract features even if cache exists.
        batch_size: Batch size for feature extraction.

    Returns:
        train_dataset: MultiEnvDataset with features from source domains.
        test_dataset: MultiEnvDataset with features from target domain.

    Raises:
        ValueError: If target_domain is not a valid PACS domain.
    """
    if target_domain not in PACS_DOMAINS:
        raise ValueError(f"target_domain must be one of {PACS_DOMAINS}, got '{target_domain}'")

    cache_dir = os.path.join(data_dir, '_features_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'pacs_resnet18_target_{target_domain}.pt')

    if os.path.exists(cache_path) and not force_extract:
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        train_dataset = MultiEnvDataset(cached['X_train'], cached['Y_train'], cached['E_train'])
        test_dataset = MultiEnvDataset(cached['X_test'], cached['Y_test'], cached['E_test'])
        return train_dataset, test_dataset

    # Extract features
    source_domains = [d for d in PACS_DOMAINS if d != target_domain]

    print(f"Extracting PACS features (target={target_domain})...")
    print(f"  Source domains: {source_domains}")
    print(f"  Loading ResNet18 feature extractor...")
    model = _get_resnet18_feature_extractor(device)

    # Source (train) features
    print(f"  Extracting source domain features...")
    source_dataset = PACSImageDataset(data_dir, source_domains)
    X_train, Y_train, E_train = _extract_features(source_dataset, model, device, batch_size)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features, "
          f"{E_train.max().item() + 1} environments")

    # Target (test) features
    print(f"  Extracting target domain features...")
    target_dataset = PACSImageDataset(data_dir, [target_domain])
    X_test, Y_test, E_test = _extract_features(target_dataset, model, device, batch_size)
    # Target domain is a single environment (env 0)
    print(f"  Test: {X_test.shape[0]} samples from '{target_domain}'")

    # Cache to disk
    torch.save({
        'X_train': X_train, 'Y_train': Y_train, 'E_train': E_train,
        'X_test': X_test, 'Y_test': Y_test, 'E_test': E_test,
        'source_domains': source_domains, 'target_domain': target_domain,
    }, cache_path)
    print(f"  Cached features to {cache_path}")

    train_dataset = MultiEnvDataset(X_train, Y_train, E_train)
    test_dataset = MultiEnvDataset(X_test, Y_test, E_test)
    return train_dataset, test_dataset


def get_all_pacs_splits(data_dir: str, device: str = 'cpu',
                        force_extract: bool = False, batch_size: int = 64
                        ) -> Dict[str, Tuple[MultiEnvDataset, MultiEnvDataset]]:
    """Get all leave-one-domain-out splits for PACS.

    Returns:
        Dictionary mapping target domain name to (train_dataset, test_dataset).
    """
    splits = {}
    for target in PACS_DOMAINS:
        splits[target] = get_pacs_features(
            data_dir, target, device=device,
            force_extract=force_extract, batch_size=batch_size
        )
    return splits
