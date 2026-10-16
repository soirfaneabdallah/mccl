# mccl/src/mccl/evaluation/calibration.py
"""Calibration : ECE et diagrammes de fiabilité."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


@torch.no_grad()
def expected_calibration_error(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    :param logits: tenseur [N, C]
    :param y: tenseur [N]
    :param n_bins: nombre de bins
    """
    p = torch.softmax(logits, dim=-1)
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float()

    conf = conf.cpu().numpy()
    correct = correct.cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)


@torch.no_grad()
def reliability_diagram(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (confs_moyennes, accuracies, tailles) par bin."""
    p = torch.softmax(logits, dim=-1)
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float()

    conf = conf.cpu().numpy()
    correct = correct.cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, sizes = [], [], []

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            confs.append(conf[mask].mean())
            accs.append(correct[mask].mean())
            sizes.append(int(mask.sum()))

    return np.array(confs), np.array(accs), np.array(sizes)


@torch.no_grad()
def brier_score(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Brier score multi-classe."""
    p = torch.softmax(logits, dim=-1)
    n, c = p.shape
    y_onehot = torch.zeros_like(p)
    y_onehot.scatter_(1, y.unsqueeze(-1), 1.0)
    return float(((p - y_onehot) ** 2).sum(dim=-1).mean().item())