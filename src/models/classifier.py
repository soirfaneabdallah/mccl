# mccl/src/mccl/models/classifier.py
"""Classifieur à prototypes extensible.

- un prototype par classe, appris sur la variété ;
- la classification se fait par distance d_M² au prototype ;
- les prototypes peuvent être ajoutés dynamiquement (tâches incrémentales).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.geometry.manifolds import ProductManifold


class PrototypeClassifier(nn.Module):
    """Classifieur à base de prototypes sur la variété.

    :param manifold: ProductManifold sur lequel vivent les prototypes
    :param temperature: température pour les logits (softmax(-d²/T))
    """

    def __init__(
        self,
        manifold: ProductManifold,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature

        # Registre : class_id -> paramètre
        self.prototypes = nn.ParameterDict()
        self._class_ids: List[int] = []

    # ---------------------- Gestion des classes ----------------------
    def add_class(
        self,
        class_id: int,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """Ajoute une nouvelle classe. Idempotent.

        :param class_id: identifiant entier de la classe
        :param init: tenseur [D] d'initialisation (par défaut : bruit léger)
        """
        key = str(class_id)
        if key in self.prototypes:
            return

        if init is None:
            # Initialisation près de l'origine dans T_0, puis expmap0
            v = torch.randn(self.manifold.dim) * 0.02
            init = self.manifold.expmap0(v.unsqueeze(0)).squeeze(0)

        self.prototypes[key] = nn.Parameter(init.clone().detach())
        self._class_ids.append(class_id)

    def add_classes(self, class_ids: List[int]) -> None:
        for c in class_ids:
            self.add_class(c)

    def n_classes(self) -> int:
        return len(self._class_ids)

    def class_ids(self) -> List[int]:
        return list(self._class_ids)

    def prototype_tensor(self) -> torch.Tensor:
        """Renvoie [C, D] des prototypes, dans l'ordre des _class_ids."""
        return torch.stack(
            [self.prototypes[str(c)] for c in self._class_ids],
            dim=0,
        )

    # ---------------------- Forward ----------------------
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Calcule les logits = -d_M²(g, prototype) / T.

        :param g: [B, D] embeddings globaux
        :return: [B, C] logits
        """
        if not self._class_ids:
            return torch.zeros(g.size(0), 0, device=g.device)

        protos = self.prototype_tensor()          # [C, D]
        protos = self.manifold.proj(protos)

        # d² : [B, C]
        d2 = self.manifold.dist_sq(
            g.unsqueeze(1),                        # [B, 1, D]
            protos.unsqueeze(0),                   # [1, C, D]
        )
        return -d2 / self.temperature

    def predict(self, g: torch.Tensor) -> torch.Tensor:
        """Renvoie l'index local (0..C-1) de la classe prédite."""
        logits = self.forward(g)
        return logits.argmax(dim=-1)

    def predict_class_id(self, g: torch.Tensor) -> List[int]:
        """Renvoie les class_id prédits."""
        idx = self.predict(g).tolist()
        return [self._class_ids[i] for i in idx]

    # ---------------------- Mise à jour ----------------------
    @torch.no_grad()
    def update_prototype(
        self,
        class_id: int,
        new_proto: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        """Met à jour un prototype par moyenne mobile."""
        key = str(class_id)
        if key not in self.prototypes:
            self.add_class(class_id, new_proto)
            return
        old = self.prototypes[key].data
        self.prototypes[key].data = momentum * old + (1 - momentum) * new_proto

    def state_dict_classes(self) -> Dict:
        """Renvoie les classes et leurs prototypes (pour sauvegarde)."""
        return {
            "class_ids": list(self._class_ids),
            "prototypes": {
                str(c): self.prototypes[str(c)].detach().cpu()
                for c in self._class_ids
            },
        }