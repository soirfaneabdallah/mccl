# mccl/src/mccl/models/predictor.py
"""Prédicteur JEPA : contexte → cibles masquées, en coordonnées tangentes.

L'entrée et la sortie sont dans l'espace tangent à l'origine (logmap0)
pour rester compatible avec les opérations MLP standard.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """MLP qui prédit les cibles masquées à partir du contexte.

    :param dim: dimension de l'espace tangent
    :param hidden: dimension cachée
    :param depth: nombre de couches cachées
    """

    def __init__(self, dim: int, hidden: int = 384, depth: int = 2) -> None:
        super().__init__()
        layers = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)

        # Initialisation
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit dans l'espace tangent.

        :param x: [B, D] dans T_0
        :return: [B, D] dans T_0
        """
        return self.net(x)


def pool_context(y_ctx: torch.Tensor, manifold) -> torch.Tensor:
    """Pooling du contexte dans T_0 (moyenne).

    :param y_ctx: [B, N_ctx, D] sur la variété
    :param manifold: ProductManifold
    :return: [B, D] dans T_0
    """
    v = manifold.logmap0(y_ctx)
    return v.mean(dim=1)