# mccl/src/mccl/utils/seed.py
"""Contrôle de la reproductibilité : seeds Python, NumPy, PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Fixe toutes les sources d'aléa pour la reproductibilité.

    :param seed: graine
    :param deterministic: si True, active les modes déterministes de
        cuDNN (plus lent mais reproductible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Pour torch >= 1.8, il faut parfois forcer :
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def worker_init_fn(worker_id: int) -> None:
    """Fonction à passer à DataLoader(num_workers>0, worker_init_fn=...).

    Assure que chaque worker a une seed différente et reproductible.
    """
    base_seed = torch.initial_seed() % (2 ** 32)
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def get_rng_state() -> dict:
    """Capture l'état des générateurs aléatoires."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    """Restaure l'état des générateurs aléatoires."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])