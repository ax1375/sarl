"""Training callbacks."""
import torch
from pathlib import Path


class Callback:
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics): pass


class EarlyStopping(Callback):
    """Early stopping callback to stop training when monitored metric stops improving.

    Args:
        monitor: Metric name to monitor ('val_loss', 'train_loss', etc.)
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change in monitored metric to qualify as improvement
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
    """
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.monitor, self.patience, self.min_delta, self.mode = monitor, patience, min_delta, mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        current = {**train_metrics, **val_metrics}.get(self.monitor)
        if current is None:
            return
        improved = (self.mode == 'min' and current < self.best - self.min_delta) or \
                   (self.mode == 'max' and current > self.best + self.min_delta)
        if improved:
            self.best, self.counter = current, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                self.should_stop = True
                trainer.stop_training = True  # Signal trainer to stop


class ModelCheckpoint(Callback):
    def __init__(self, path: str = 'checkpoints', monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.monitor, self.mode, self.save_best_only = monitor, mode, save_best_only
        self.best = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        current = {**train_metrics, **val_metrics}.get(self.monitor, train_metrics.get('loss'))
        improved = (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best)
        if improved or not self.save_best_only:
            if improved:
                self.best = current
            trainer.save(str(self.path / f'checkpoint_epoch{epoch}.pt'))


class StructureMonitor(Callback):
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        if epoch % self.log_every == 0:
            print(f"  Structure: G{trainer.objective.get_predicted_structure()}, "
                  f"V=({train_metrics.get('v1',0):.4f}, {train_metrics.get('v2',0):.4f}, {train_metrics.get('v3',0):.4f})")
