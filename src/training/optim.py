# mccl/src/mccl/training/optim.py
"""Optimiseurs : Adam/AdamW pour les paramètres euclidiens,
étape riemannienne optionnelle pour les paramètres sur variété.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Optimiseur standard
# ---------------------------------------------------------------------------
def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    hyperbolic_lr_scale: float = 0.1,
) -> torch.optim.Optimizer:
    """Construit un AdamW avec des groupes de paramètres distincts.

    Les paramètres "hyperboliques" (prototypes, biais près du bord)
    utilisent un learning rate plus faible pour éviter les instabilités.

    :param model: modèle PyTorch
    :param lr: learning rate de base
    :param weight_decay: régularisation L2
    :param hyperbolic_lr_scale: multiplicateur de lr pour les prototypes
    """
    proto_params = []
    proto_ids = set()

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if hasattr(classifier, "prototypes"):
            for p in classifier.prototypes.parameters():
                proto_params.append(p)
                proto_ids.add(id(p))

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in proto_ids
    ]

    groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": proto_params, "lr": lr * hyperbolic_lr_scale, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]

    return torch.optim.AdamW(groups, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    min_lr_ratio: float = 0.01,
):
    """Scheduler cosinus avec warmup linéaire."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cos_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor.item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Étape riemannienne (expérimentale)
# ---------------------------------------------------------------------------
@torch.no_grad()
def riemannian_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    manifold,
) -> None:
    """Met à jour un paramètre sur la variété par gradient riemannien.

    Formule simplifiée (RSGD) :
        x_{t+1} = exp_x(-lr * grad_euclidien)

    Note : le gradient doit être le gradient euclidien par rapport à x,
    pas le gradient dans T_x. Cette étape approxime en réutilisant
    logmap0/expmap0 au point courant.

    :param param: tenseur de paramètre sur la variété
    :param grad: gradient euclidien de même forme
    :param lr: learning rate
    :param manifold: ProductManifold
    """
    # Projection sur la variété au cas où
    x = manifold.proj(param)
    # Descente dans l'espace tangent à x, exprimée à partir de 0
    # (approximation : on reste sur la paramétrisation ambient)
    v = manifold.logmap0(x)
    v = v - lr * grad
    param.copy_(manifold.expmap0(v))


# ---------------------------------------------------------------------------
# Clip et stabilité
# ---------------------------------------------------------------------------
def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """Clip des gradients, renvoie la norme avant clip."""
    return float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm,
    ))


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    factor: float,
) -> None:
    """Multiplie le lr de tous les groupes par un facteur."""
    for group in optimizer.param_groups:
        group["lr"] *= factor