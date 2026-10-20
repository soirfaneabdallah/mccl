# mccl/src/mccl/models/jepa.py
"""Modèle JEPA complet : backbone + tête variété + prédicteur + classifieur."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .backbone import DinoV2Backbone
from .manifold_head import ManifoldHead
from .predictor import Predictor, pool_context
from .classifier import PrototypeClassifier


@dataclass
class JepaOutput:
    """Sortie du modèle JEPA."""
    tokens: torch.Tensor                # [B, N, D_man]
    global_embed: torch.Tensor          # [B, D_man]
    logits: Optional[torch.Tensor]      # [B, C] ou None
    pred: Optional[torch.Tensor]        # [B, D_man] dans T_0
    target: Optional[torch.Tensor]      # [B, D_man] dans T_0


class JepaModel(nn.Module):
    """Assemble backbone, tête variété, prédicteur et classifieur.

    :param model_cfg: objet avec .backbone, .manifold, .predictor, .classifier
    """

    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.cfg = model_cfg

        # Backbone
        self.backbone = DinoV2Backbone(
            name=model_cfg.backbone.name,
            num_blocks=model_cfg.backbone.num_blocks,
            frozen=model_cfg.backbone.frozen,
        )

        # Tête variété
        self.head = ManifoldHead(
            in_dim=self.backbone.embed_dim,
            manifold_cfg=model_cfg.manifold,
            pooling=model_cfg.manifold.pooling,
        )

        # Prédicteur JEPA
        self.predictor = Predictor(
            dim=self.head.manifold.dim,
            hidden=model_cfg.predictor.hidden,
            depth=model_cfg.predictor.depth,
        )

        # Classifieur à prototypes
        self.classifier = PrototypeClassifier(self.head.manifold)

    @property
    def manifold(self):
        return self.head.manifold

    def forward(
        self,
        x: torch.Tensor,
        ctx_idx: Optional[torch.Tensor] = None,
        tgt_idx: Optional[torch.Tensor] = None,
    ) -> JepaOutput:
        """Forward complet.

        :param x: [B, 3, H, W]
        :param ctx_idx: indices des patchs contexte (JEPA)
        :param tgt_idx: indices des patchs cibles (JEPA)
        """
        # 1. Backbone (features tokens)
        tokens_bb = self.backbone(x)               # [B, N, D_bb]

        # 2. Tête variété
        tokens_man, g = self.head(tokens_bb)       # [B, N, D_man], [B, D_man]

        # 3. Classifieur (si au moins une classe)
        logits = None
        if self.classifier.n_classes() > 0:
            logits = self.classifier(g)            # [B, C]

        # 4. Prédiction JEPA (optionnelle)
        pred = target = None
        if ctx_idx is not None and tgt_idx is not None:
            ctx_idx = ctx_idx.to(x.device)
            tgt_idx = tgt_idx.to(x.device)

            # Contexte
            y_ctx = tokens_man.index_select(1, ctx_idx)   # [B, N_ctx, D]
            v_ctx = pool_context(y_ctx, self.manifold)    # [B, D] dans T_0
            pred = self.predictor(v_ctx)                  # [B, D] dans T_0

            # Cible
            y_tgt = tokens_man.index_select(1, tgt_idx)   # [B, N_tgt, D]
            v_tgt = pool_context(y_tgt, self.manifold)    # [B, D] dans T_0
            target = v_tgt

        return JepaOutput(
            tokens=tokens_man,
            global_embed=g,
            logits=logits,
            pred=pred,
            target=target,
        )

    # ---------------------- Utilitaires ----------------------
    def register_new_classes(self, class_ids) -> None:
        """Enregistre de nouvelles classes dans le classifieur."""
        for c in class_ids:
            self.classifier.add_class(int(c))

    def trainable_parameters(self):
        """Renvoie les paramètres entraînables (backbone gelé exclu)."""
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def count_parameters(self) -> dict:
        """Compte les paramètres totaux et entraînables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}