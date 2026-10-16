# mccl/src/mccl/data/masking.py
"""Génération de masques contexte/cible pour le JEPA masqué."""
from __future__ import annotations

from typing import Tuple

import torch


def random_block_mask(
    grid_h: int,
    grid_w: int,
    n_context: int,
    n_target: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Renvoie (context_idx, target_idx) sur une grille aplatie.

    :param grid_h: hauteur de la grille de patchs (ex. 16 pour 224/14)
    :param grid_w: largeur de la grille
    :param n_context: nombre de patchs contexte
    :param n_target: nombre de patchs cible
    :param generator: générateur torch pour la reproductibilité
    :return: deux tenseurs d'indices
    """
    n = grid_h * grid_w
    if n_context + n_target > n:
        raise ValueError(
            f"n_context + n_target ({n_context + n_target}) > n ({n})"
        )
    perm = torch.randperm(n, generator=generator)
    ctx = perm[:n_context]
    tgt = perm[n_context:n_context + n_target]
    return ctx, tgt


def apply_mask(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Sélectionne les tokens aux indices donnés.

    :param tokens: tenseur [B, N, D]
    :param idx: tenseur d'indices [K]
    :return: tenseur [B, K, D]
    """
    return tokens.index_select(1, idx.to(tokens.device))