# mccl/src/mccl/models/backbone.py
"""Backbone DINOv2 tronqué et gelable.

On garde les `num_blocks` premiers blocs (par défaut 10 sur 12 pour
DINOv2-small). Le tronc peut être gelé pour éviter la dérive et
permettre la mise en cache des features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.logging import get_logger

log = get_logger(__name__)


class DinoV2Backbone(nn.Module):
    """DINOv2 tronqué, avec sortie de tokens.

    :param name: nom du modèle sur torch.hub (ex. "dinov2-small")
    :param num_blocks: nombre de blocs Transformer à conserver
    :param frozen: si True, tous les paramètres sont gelés
    """

    def __init__(
        self,
        name: str = "dinov2-small",
        num_blocks: int = 10,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_blocks = num_blocks
        self.frozen = frozen

        try:
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception as e:
            raise RuntimeError(
                f"Impossible de charger {name} via torch.hub : {e}"
            ) from e

        # Troncature des blocs
        if hasattr(self.net, "blocks"):
            all_blocks = list(self.net.blocks)
            if num_blocks < len(all_blocks):
                self.net.blocks = nn.ModuleList(all_blocks[:num_blocks])
                log.info(
                    "Backbone tronqué : %d blocs sur %d",
                    num_blocks, len(all_blocks),
                )

        # Dimension d'embedding
        self.embed_dim = int(getattr(self.net, "embed_dim", 384))

        # Gel
        if frozen:
            for p in self.net.parameters():
                p.requires_grad = False
            log.info("Backbone gelé")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie les tokens de patch [B, N, D].

        :param x: tenseur [B, 3, H, W]
        :return: tenseur [B, N, D] (tokens patch, sans le CLS)
        """
        feats = self.net.forward_features(x)

        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens")
            if tokens is None:
                tokens = feats.get("x")
        else:
            tokens = feats

        if tokens is None:
            raise RuntimeError("Impossible d'extraire les tokens du backbone.")
        return tokens

    def freeze(self) -> None:
        """Gèle tous les paramètres."""
        for p in self.net.parameters():
            p.requires_grad = False
        self.frozen = True

    def unfreeze(self) -> None:
        """Dégèle tous les paramètres."""
        for p in self.net.parameters():
            p.requires_grad = True
        self.frozen = False