"""Visualization utilities."""
from typing import Dict, Optional


def plot_violations(history: Dict, save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(history['violations']['v1']))
    ax.plot(epochs, history['violations']['v1'], label='V1', color='red')
    ax.plot(epochs, history['violations']['v2'], label='V2', color='blue')
    ax.plot(epochs, history['violations']['v3'], label='V3', color='green')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Violation'); ax.legend(); ax.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_weights(history: Dict, save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(history['weights']['alpha1']))
    ax.plot(epochs, history['weights']['alpha1'], label='α1', color='red')
    ax.plot(epochs, history['weights']['alpha2'], label='α2', color='blue')
    ax.plot(epochs, history['weights']['alpha3'], label='α3', color='green')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Weight'); ax.legend(); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(len(history['train_loss']))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    if history.get('val_loss'): axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if history.get('structure_id'):
        axes[1].plot(epochs, history['structure_id'], 'o-', markersize=2)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Structure'); axes[1].set_yticks([1,2,3])
        axes[1].set_yticklabels(['G1','G2','G3']); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
