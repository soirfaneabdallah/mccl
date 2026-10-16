# mccl/src/mccl/evaluation/metrics.py
"""Métriques de classification : accuracy, par classe, matrice de confusion."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy globale.

    :param logits: tenseur [N, C]
    :param y: tenseur [N] d'entiers
    """
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def per_class_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
) -> np.ndarray:
    """Accuracy par classe.

    :return: array de taille [n_classes], NaN si classe absente
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    out = np.full(n_classes, np.nan, dtype=float)
    for c in range(n_classes):
        mask = y_np == c
        if mask.any():
            out[c] = float((pred[mask] == y_np[mask]).mean())
    return out


@torch.no_grad()
def confusion_matrix(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
) -> np.ndarray:
    """Matrice de confusion [n_classes, n_classes] (lignes = vrais)."""
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_np, pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
    """Accuracy top-k."""
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == y.unsqueeze(-1)).any(dim=-1)
    return float(correct.float().mean().item())


def per_class_counts(y: torch.Tensor, n_classes: int) -> Dict[int, int]:
    """Nombre d'échantillons par classe."""
    y_np = y.cpu().numpy()
    counts = np.bincount(y_np, minlength=n_classes)
    return {c: int(counts[c]) for c in range(n_classes)}