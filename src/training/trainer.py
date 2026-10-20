# mccl/src/mccl/training/trainer.py
"""Boucle d'entraînement pour une tâche incrémentale.

Améliorations :
- barres tqdm imbriquées (époques × batches) ;
- postfix live : loss courante, accuracy glissante, lr ;
- barre dédiée pour la validation ;
- options verbose pour désactiver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# État d'entraînement
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    """État courant de l'entraîneur."""
    task: int = 0
    epoch: int = 0
    step: int = 0
    best_val_acc: float = 0.0
    history: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def train_one_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    n_epochs: int = 1,
    loss_fn: Optional[Callable] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_grad_norm: float = 1.0,
    log_every: int = 50,
    on_step: Optional[Callable] = None,
    on_epoch: Optional[Callable] = None,
    task_idx: int = 0,
    verbose: bool = True,
) -> TrainState:
    """Entraîne le modèle sur une tâche (plusieurs époques).

    :param model: modèle à entraîner
    :param train_loader: DataLoader d'entraînement
    :param val_loader: DataLoader de validation (optionnel)
    :param optimizer: optimiseur
    :param scheduler: scheduler (optionnel)
    :param n_epochs: nombre d'époques
    :param loss_fn: fonction de perte (model, batch, out) -> loss
    :param device: 'cuda' ou 'cpu'
    :param max_grad_norm: clip des gradients
    :param log_every: fréquence de log (non utilisé si verbose=False)
    :param on_step: callback(state, batch) après chaque step
    :param on_epoch: callback(state, val_acc) après chaque époque
    :param task_idx: index de la tâche (pour l'affichage)
    :param verbose: afficher les barres tqdm
    :return: TrainState final
    """
    state = TrainState(task=task_idx)
    model.train()

    epoch_iter = range(n_epochs)
    if verbose:
        epoch_iter = tqdm(
            epoch_iter,
            desc=f"[Tâche {task_idx}] époques",
            position=0,
            leave=True,
        )

    for epoch in epoch_iter:
        state.epoch = epoch
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0

        batch_iter = train_loader
        if verbose:
            batch_iter = tqdm(
                train_loader,
                desc=f"  epoch {epoch+1}/{n_epochs}",
                position=1,
                leave=False,
            )

        for batch in batch_iter:
            # --- Préparer le batch ---
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
            else:
                x = batch.to(device, non_blocking=True)
                y = None

            # --- Forward ---
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out

            # --- Perte ---
            if loss_fn is not None:
                loss = loss_fn(model, batch, out)
            else:
                if y is None:
                    raise ValueError("Pas de labels dans le batch et pas de loss_fn")
                loss = torch.nn.functional.cross_entropy(logits, y)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # --- Statistiques ---
            state.step += 1
            epoch_loss += float(loss.item())
            n_batches += 1

            if y is not None and logits is not None:
                pred = logits.argmax(dim=-1)
                n_correct += (pred == y).sum().item()
                n_total += y.size(0)

            # --- Postfix live ---
            if verbose and n_batches % log_every == 0:
                avg_loss = epoch_loss / n_batches
                avg_acc = n_correct / max(n_total, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                batch_iter.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.3f}",
                    "lr": f"{lr_now:.2e}",
                })

            if on_step is not None:
                on_step(state, batch)

        # --- Fin d'époque ---
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = n_correct / max(n_total, 1)
        log.info("[Tâche %d] époque %d | loss=%.4f | train_acc=%.4f",
                 task_idx, epoch, avg_loss, avg_acc)

        # --- Validation ---
        entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
        }

        if val_loader is not None:
            val_acc = evaluate_accuracy(
                model, val_loader, device, verbose=verbose,
            )
            state.best_val_acc = max(state.best_val_acc, val_acc)
            entry["val_acc"] = val_acc
            log.info("[Tâche %d] val_acc = %.4f (best = %.4f)",
                     task_idx, val_acc, state.best_val_acc)
            if on_epoch is not None:
                on_epoch(state, val_acc)

        state.history.append(entry)

        # Mise à jour du postfix de la barre d'époques
        if verbose:
            epoch_iter.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.3f}",
            })

    return state


# ---------------------------------------------------------------------------
# Évaluation rapide (accuracy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    verbose: bool = False,
) -> float:
    """Calcule l'accuracy sur un loader."""
    model.eval()
    correct = 0
    total = 0

    iterable = loader
    if verbose:
        iterable = tqdm(loader, desc="  validation", position=1, leave=False)

    for batch in iterable:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            y = None
        if y is None:
            continue

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Trainer orienté objet
# ---------------------------------------------------------------------------
class Trainer:
    """Encapsule l'optimiseur, le scheduler et l'état."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_fn: Optional[Callable] = None,
        max_grad_norm: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.state = TrainState()

    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 1,
    ) -> TrainState:
        state = train_one_task(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            n_epochs=n_epochs,
            loss_fn=self.loss_fn,
            device=self.device,
            max_grad_norm=self.max_grad_norm,
            task_idx=self.state.task + 1,
            verbose=self.verbose,
        )
        self.state = state
        return state

    def save(self, path: str) -> None:
        """Sauvegarde modèle et optimiseur."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "state": self.state.__dict__,
        }, path)
        log.info("Checkpoint sauvegardé : %s", path)

    def load(self, path: str) -> None:
        """Charge modèle et optimiseur."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.state = TrainState(**ckpt.get("state", {}))
        log.info("Checkpoint chargé : %s", path)