"""Evaluation metrics."""
import torch
from typing import Dict


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    pred_labels = predictions.argmax(dim=1) if predictions.dim() > 1 and predictions.shape[1] > 1 else (predictions.squeeze() > 0).long()
    return (pred_labels == labels).float().mean().item()


def per_env_accuracy(predictions: torch.Tensor, labels: torch.Tensor, envs: torch.Tensor) -> Dict[int, float]:
    return {int(e): accuracy(predictions[envs == e], labels[envs == e]) for e in envs.unique()}
