# mccl/src/mccl/training/__init__.py
"""Entraînement : pertes, optimiseurs, boucle d'entraînement."""

from .losses import (
    jepa_loss,
    jepa_multi_step,
    classification_loss,
    reg_loss,
    total_loss,
)
from .optim import build_optimizer, build_scheduler, riemannian_step
from .trainer import Trainer, TrainState, train_one_task

__all__ = [
    "jepa_loss", "jepa_multi_step", "classification_loss",
    "reg_loss", "total_loss",
    "build_optimizer", "build_scheduler", "riemannian_step",
    "Trainer", "TrainState", "train_one_task",
]