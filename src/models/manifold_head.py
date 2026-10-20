# mccl/src/mccl/models/manifold_head.py
"""Tête variété : projette les tokens du backbone sur un ProductManifold.

Le pooling global se fait dans l'espace tangent à l'origine, puis expmap0
pour revenir sur la variété.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.geometry.manifolds import ProductManifold, build_manifold


class ManifoldHead(nn.Module):
    """Projette les features du backbone sur une variété produit.

    :param in_dim: dimension d'entrée (embed_dim du backbone)
    :param manifold_cfg: config ManifoldCfg avec `.components` et `.pooling`
    :param pooling: 'tangent_mean' | 'mean' | 'cls'
    """

    def __init__(
        self,
        in_dim: int,
        manifold_cfg,
        pooling: str = "tangent_mean",
    ) -> None:
        super().__init__()
        self.manifold: ProductManifold = build_manifold(manifold_cfg)
        self.pooling = pooling

        # Projection linéaire vers l'espace ambient de la variété
        self.proj = nn.Linear(in_dim, self.manifold.dim)

        # Initialisation : petites valeurs pour rester près de l'origine
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def pool(self, tokens_manifold: torch.Tensor) -> torch.Tensor:
        """Pooling global sur la variété.

        :param tokens_manifold: [B, N, D] sur la variété
        :return: [B, D] sur la variété
        """
        if self.pooling == "mean":
            # Moyenne euclidienne (uniquement valide pour E)
            v = tokens_manifold.mean(dim=1)
            return self.manifold.proj(v)

        if self.pooling == "cls":
            return tokens_manifold[:, 0]

        # tangent_mean : moyenne dans T_0, puis expmap0
        v = self.manifold.logmap0(tokens_manifold)
        v = v.mean(dim=1)
        return self.manifold.expmap0(v)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projette les tokens et renvoie (tokens_manifold, global_embed).

        :param tokens: [B, N, D_in] du backbone
        :return: y [B, N, D_man], g [B, D_man]
        """
        v = self.proj(tokens)                     # [B, N, D_man]
        v = v.clamp(-10.0, 10.0)                  # évite les explosions
        y = self.manifold.expmap0(v)              # [B, N, D_man]
        y = self.manifold.proj(y)
        g = self.pool(y)                          # [B, D_man]
        return y, g