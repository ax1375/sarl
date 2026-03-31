"""Main SARL trainer."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from ..losses import SARLObjective, TemperatureScheduler
from ..models import SARLModel


class SARLTrainer:
    """Trainer for Structure-Agnostic Representation Learning."""
    def __init__(self, model: SARLModel, objective: SARLObjective, optimizer: Optional[optim.Optimizer] = None,
                 device: str = 'auto', beta_scheduler: Optional[TemperatureScheduler] = None, lr: float = 1e-4):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
        self.model = model.to(self.device)
        self.objective = objective.to(self.device)
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        self.beta_scheduler = beta_scheduler
        self.epoch, self.global_step = 0, 0
        self.stop_training = False  # Flag for early stopping
        self.history = {'train_loss': [], 'val_loss': [], 'violations': {'v1': [], 'v2': [], 'v3': []},
                       'weights': {'alpha1': [], 'alpha2': [], 'alpha3': []}, 'beta': [], 'structure_id': []}
        self.callbacks = []
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def train_epoch(self, train_loader: DataLoader, loss_fn: Optional[nn.Module] = None) -> Dict[str, float]:
        self.model.train()
        epoch_loss, n_batches = 0.0, 0
        epoch_components = {'pred_loss': 0., 'inv_penalty': 0., 'v1': 0., 'v2': 0., 'v3': 0.}
        
        for X, Y, E in train_loader:
            X, Y, E = X.to(self.device), Y.to(self.device), E.to(self.device)
            phi_x, predictions = self.model.forward_with_representation(X)
            loss_dict = self.objective(phi_x, Y, E, predictions, loss_fn, return_components=True)
            
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss_dict['total'].item()
            for k in epoch_components:
                if k in loss_dict:
                    epoch_components[k] += loss_dict[k].item()
            n_batches += 1
            self.global_step += 1
        
        return {'loss': epoch_loss / n_batches, **{k: v / n_batches for k, v in epoch_components.items()}}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, loss_fn: Optional[nn.Module] = None) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0., 0, 0
        all_phi, all_y, all_e, all_preds = [], [], [], []

        for X, Y, E in loader:
            X, Y, E = X.to(self.device), Y.to(self.device), E.to(self.device)
            phi_x, predictions = self.model.forward_with_representation(X)
            loss_dict = self.objective(phi_x, Y, E, predictions, loss_fn, return_components=True)
            total_loss += loss_dict['total'].item() * len(X)
            if Y.dtype in [torch.long, torch.int]:
                pred_labels = predictions.argmax(dim=-1) if predictions.dim() > 1 else (predictions > 0).long()
                total_correct += (pred_labels == Y).sum().item()
            total_samples += len(X)
            all_phi.append(phi_x.cpu())
            all_y.append(Y.cpu())
            all_e.append(E.cpu())
            all_preds.append(predictions.cpu())

        phi_all = torch.cat(all_phi).to(self.device)
        y_all = torch.cat(all_y).to(self.device)
        e_all = torch.cat(all_e).to(self.device)
        preds_all = torch.cat(all_preds).to(self.device)
        violations = self.objective.compute_violations_only(phi_all, y_all, e_all, predictions=preds_all)
        
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples,
                'v1': violations.v1.item(), 'v2': violations.v2.item(), 'v3': violations.v3.item()}
    
    def identify_structure(self, loader: DataLoader) -> Tuple[int, Dict[str, float]]:
        self.model.eval()
        all_phi, all_y, all_e, all_preds = [], [], [], []
        with torch.no_grad():
            for X, Y, E in loader:
                X_dev = X.to(self.device)
                phi, pred = self.model.forward_with_representation(X_dev)
                all_phi.append(phi.cpu())
                all_preds.append(pred.cpu())
                all_y.append(Y)
                all_e.append(E)
        phi_all = torch.cat(all_phi).to(self.device)
        y_all = torch.cat(all_y).to(self.device)
        e_all = torch.cat(all_e).to(self.device)
        preds_all = torch.cat(all_preds).to(self.device)
        return self.objective.identify_structure(phi_all, y_all, e_all, predictions=preds_all)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, n_epochs: int = 100,
              loss_fn: Optional[nn.Module] = None, verbose: bool = True) -> Dict[str, List]:
        for cb in self.callbacks:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin(self)
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            if self.beta_scheduler:
                beta = self.beta_scheduler.step(self.objective.adaptive_weights.softmin, epoch)
                self.history['beta'].append(beta)
            
            train_metrics = self.train_epoch(train_loader, loss_fn)
            self.history['train_loss'].append(train_metrics['loss'])
            for k in ['v1', 'v2', 'v3']:
                self.history['violations'][k].append(train_metrics.get(k, 0))
            
            weights = self.objective.get_structure_weights()
            if weights is not None:
                for i, k in enumerate(['alpha1', 'alpha2', 'alpha3']):
                    self.history['weights'][k].append(weights[i].item() if torch.is_tensor(weights[i]) else weights[i])
            
            val_metrics = self.evaluate(val_loader, loss_fn) if val_loader else {}
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
            
            self.history['structure_id'].append(self.objective.get_predicted_structure())
            
            for cb in self.callbacks:
                if hasattr(cb, 'on_epoch_end'):
                    cb.on_epoch_end(self, epoch, train_metrics, val_metrics)

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}: loss={train_metrics['loss']:.4f}"
                if val_metrics:
                    msg += f", val_acc={val_metrics.get('accuracy', 0):.4f}"
                msg += f", struct={self.history['structure_id'][-1]}"
                print(msg)

            # Check for early stopping
            if self.stop_training:
                if verbose:
                    print(f"Training stopped early at epoch {epoch}")
                break

        return self.history
    
    def save(self, path: str):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'epoch': self.epoch, 'history': self.history}, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epoch, self.history = ckpt['epoch'], ckpt['history']


def train_sarl(model: SARLModel, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
               n_epochs: int = 100, lambda_inv: float = 1.0, beta_start: float = 1.0, beta_end: float = 50.0,
               lr: float = 1e-4, device: str = 'auto', verbose: bool = True) -> Tuple['SARLTrainer', Dict]:
    """Convenience function for training SARL."""
    objective = SARLObjective(lambda_inv=lambda_inv, beta=beta_start, representation_dim=model.representation_dim,
                              num_envs=train_loader.dataset.n_envs)
    beta_scheduler = TemperatureScheduler(beta_start, beta_end, anneal_fraction=0.5, total_epochs=n_epochs)
    trainer = SARLTrainer(model, objective, device=device, beta_scheduler=beta_scheduler, lr=lr)
    history = trainer.train(train_loader, val_loader, n_epochs, verbose=verbose)
    return trainer, history


def train_with_restarts(model_fn, train_loader: DataLoader, val_loader: DataLoader,
                        n_restarts: int = 5, n_epochs: int = 100, lambda_inv: float = 1.0,
                        beta_start: float = 1.0, beta_end: float = 50.0,
                        lr: float = 1e-4, rho: float = 0.5, device: str = 'auto',
                        verbose: bool = False) -> Tuple['SARLTrainer', Dict]:
    """Multi-start training: run n_restarts initializations, select best on val.

    Paper Section 4: "We run the optimization from M >= 10 random initializations
    and select the solution with the lowest objective value."

    Args:
        model_fn: Callable that returns a fresh SARLModel (called per restart).
        train_loader: Training data loader.
        val_loader: Validation data loader for model selection.
        n_restarts: Number of random restarts.
        n_epochs: Epochs per restart.

    Returns:
        Best trainer and its history.
    """
    best_trainer, best_val_loss, best_history = None, float('inf'), None

    for restart in range(n_restarts):
        model = model_fn()
        num_envs = train_loader.dataset.n_envs
        objective = SARLObjective(lambda_inv=lambda_inv, beta=beta_start, rho=rho,
                                  representation_dim=model.representation_dim, num_envs=num_envs)
        beta_scheduler = TemperatureScheduler(beta_start, beta_end, anneal_fraction=0.5,
                                              total_epochs=n_epochs)
        trainer = SARLTrainer(model, objective, device=device, beta_scheduler=beta_scheduler, lr=lr)
        history = trainer.train(train_loader, n_epochs=n_epochs, verbose=False)

        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics['loss']

        if verbose:
            struct = trainer.objective.get_predicted_structure()
            print(f"  restart {restart}: val_loss={val_loss:.4f}, val_acc={val_metrics['accuracy']:.1%}, struct=G{struct}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_trainer = trainer
            best_history = history

    return best_trainer, best_history
