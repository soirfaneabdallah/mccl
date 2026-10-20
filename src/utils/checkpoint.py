# mccl/src/mccl/utils/checkpoint.py
"""Sauvegarde et chargement : checkpoints PyTorch + JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoints PyTorch
# ---------------------------------------------------------------------------
def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Sauvegarde un checkpoint complet.

    :param path: chemin du fichier .pt
    :param model: modèle PyTorch
    :param optimizer: optimiseur (optionnel)
    :param scheduler: scheduler (optionnel)
    :param extra: dictionnaire d'informations supplémentaires
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optim"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    log.info("Checkpoint sauvegardé : %s", path)


def load_ckpt(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Charge un checkpoint.

    :return: dictionnaire des métadonnées (hors model/optim/scheduler)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    meta = {k: v for k, v in ckpt.items()
            if k not in ("model", "optim", "scheduler")}
    log.info("Checkpoint chargé : %s", path)
    return meta


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def save_json(path: str, data: Any, indent: int = 2) -> None:
    """Écrit un objet Python en JSON.

    Convertit les types non sérialisables : numpy, torch, tuple, etc.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=indent)


def load_json(path: str) -> Any:
    """Lit un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_serializable(obj: Any) -> Any:
    """Rend un objet Python sérialisable en JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj