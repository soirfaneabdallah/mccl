# mccl/src/mccl/continual/consolidation.py
"""Consolidation : transfert des éléments surprenants vers la mémoire lente.

Déclenché par un seuil de surprise (par défaut, quantile glissant).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .memory import FastMemory, SlowMemory
from .surprise import SurpriseScorer


class Consolidator:
    """Décide quels éléments consolider, et effectue le transfert.

    Fonctionnement :
    1. Reçoit un batch de features et un score de surprise.
    2. Sélectionne les éléments dont la surprise dépasse un seuil.
    3. Les transfère vers SlowMemory (prototypes + buffer).
    """

    def __init__(
        self,
        threshold_quantile: float = 0.9,
        min_buffer: int = 8,
    ) -> None:
        self.threshold_quantile = threshold_quantile
        self.min_buffer = min_buffer

        self.buffer_feats: List[torch.Tensor] = []
        self.buffer_labels: List[int] = []
        self.n_consolidated = 0

    # ------------------------------------------------------------------
    def maybe_store(
        self,
        surprise: torch.Tensor,
        feats: torch.Tensor,
        labels: torch.Tensor,
        slow_memory: Optional[SlowMemory] = None,
    ) -> int:
        """Sélectionne les éléments surprenants et les consolide.

        :param surprise: [B] scores de surprise normalisés
        :param feats: [B, D] features correspondantes
        :param labels: [B] labels
        :param slow_memory: si fourni, mise à jour immédiate
        :return: nombre d'éléments consolidés
        """
        if surprise.numel() == 0 or feats.numel() == 0:
            return 0

        surprise = surprise.detach().cpu()
        feats = feats.detach().cpu()
        labels = labels.detach().cpu()

        # Seuil : quantile du batch courant
        if surprise.numel() >= self.min_buffer:
            thr = torch.quantile(surprise, self.threshold_quantile).item()
        else:
            thr = surprise.mean().item()

        mask = surprise >= thr
        n_selected = int(mask.sum().item())
        if n_selected == 0:
            return 0

        sel_feats = feats[mask]
        sel_labels = labels[mask]

        # Consolidation immédiate si SlowMemory fournie
        if slow_memory is not None:
            for c in sel_labels.unique().tolist():
                m = sel_labels == c
                slow_memory.update(int(c), sel_feats[m])

        # Sinon, buffer pour traitement ultérieur
        else:
            self.buffer_feats.append(sel_feats)
            self.buffer_labels.extend(sel_labels.tolist())

        self.n_consolidated += n_selected
        return n_selected

    # ------------------------------------------------------------------
    def drain(self) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        """Vide le buffer et renvoie (features, labels)."""
        if not self.buffer_feats:
            return None, None
        feats = torch.cat(self.buffer_feats, dim=0)
        labels = list(self.buffer_labels)
        self.buffer_feats = []
        self.buffer_labels = []
        return feats, labels

    def stats(self) -> dict:
        return {
            "n_consolidated": self.n_consolidated,
            "buffer_size": sum(f.size(0) for f in self.buffer_feats),
        }