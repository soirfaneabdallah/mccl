# mccl/src/mccl/continual/memory.py
"""Mémoires de rejeu.

- FastMemory : réservoir borné, rejeu récent ou échantillonné uniformément.
- SlowMemory : prototypes par classe, mis à jour par moyenne mobile ou herding.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# FastMemory : réservoir
# ---------------------------------------------------------------------------
class FastMemory:
    """Réservoir borné pour rejeu.

    Deux stratégies :
    - 'reservoir' : remplacement aléatoire (échantillonnage uniforme).
    - 'fifo'      : file d'attente, les plus anciens sortent d'abord.

    Stocke des tenseurs sur CPU pour économiser la mémoire GPU.
    """

    def __init__(
        self,
        capacity: int = 256,
        strategy: str = "reservoir",
        store_features: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.strategy = strategy
        self.store_features = store_features

        self._x: List[torch.Tensor] = []
        self._y: List[int] = []
        self._f: List[torch.Tensor] = []        # features, si demandé
        self._n_seen = 0

    # ------------------------------------------------------------------
    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
    ) -> None:
        """Ajoute un batch à la mémoire.

        :param x: [B, ...] tenseur des entrées
        :param y: [B] labels
        :param feats: [B, D] features optionnelles
        """
        x = x.detach().cpu()
        y = y.detach().cpu()
        f = feats.detach().cpu() if feats is not None else None

        for i in range(x.size(0)):
            self._n_seen += 1
            if len(self._x) < self.capacity:
                self._x.append(x[i])
                self._y.append(int(y[i]))
                if f is not None:
                    self._f.append(f[i])
            elif self.strategy == "fifo":
                # Remplace le plus ancien
                self._x.pop(0); self._y.pop(0)
                if f is not None and self._f:
                    self._f.pop(0)
                self._x.append(x[i]); self._y.append(int(y[i]))
                if f is not None:
                    self._f.append(f[i])
            else:
                # Réservoir : remplace un élément aléatoire avec proba k/n
                j = torch.randint(0, self._n_seen, (1,)).item()
                if j < self.capacity:
                    self._x[j] = x[i]
                    self._y[j] = int(y[i])
                    if f is not None and j < len(self._f):
                        self._f[j] = f[i]

    # ------------------------------------------------------------------
    def sample(
        self,
        n: int,
        return_features: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Échantillonne n éléments (avec remise si n > taille)."""
        if not self._x:
            return None, None, None

        idx = torch.randint(0, len(self._x), (n,))
        x = torch.stack([self._x[i] for i in idx], dim=0)
        y = torch.tensor([self._y[i] for i in idx], dtype=torch.long)
        f = None
        if return_features and self._f:
            f = torch.stack([self._f[i] for i in idx], dim=0)
        return x, y, f

    def get_all(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._x:
            return None, None
        return (
            torch.stack(self._x, dim=0),
            torch.tensor(self._y, dtype=torch.long),
        )

    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for c in self._y:
            counts[c] = counts.get(c, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._x)

    def __repr__(self) -> str:
        return (f"FastMemory(capacity={self.capacity}, "
                f"size={len(self._x)}, strategy={self.strategy})")


# ---------------------------------------------------------------------------
# SlowMemory : prototypes par classe
# ---------------------------------------------------------------------------
class SlowMemory:
    """Prototypes par classe, mis à jour par moyenne mobile.

    Utilisée pour :
    - consolider les connaissances sans stocker toutes les images ;
    - fournir un signal de rejeu en espace de features.
    """

    def __init__(
        self,
        capacity_per_class: int = 32,
        momentum: float = 0.9,
    ) -> None:
        self.capacity_per_class = int(capacity_per_class)
        self.momentum = momentum

        # class_id -> tenseur [capacity, D]
        self.data: Dict[int, torch.Tensor] = {}
        # class_id -> prototype [D]
        self.prototypes: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def update(self, class_id: int, feats: torch.Tensor) -> None:
        """Met à jour le prototype et le buffer d'une classe.

        :param class_id: identifiant de classe
        :param feats: [N, D] features de cette classe
        """
        feats = feats.detach().cpu()
        if feats.numel() == 0:
            return

        # Moyenne
        mean = feats.mean(dim=0)

        # Mise à jour du prototype par moyenne mobile
        if class_id in self.prototypes:
            old = self.prototypes[class_id]
            self.prototypes[class_id] = self.momentum * old + (1 - self.momentum) * mean
        else:
            self.prototypes[class_id] = mean

        # Buffer
        old_buf = self.data.get(class_id, torch.empty(0, feats.size(-1)))
        merged = torch.cat([old_buf, feats], dim=0)
        if merged.size(0) > self.capacity_per_class:
            # Garde les plus récents
            merged = merged[-self.capacity_per_class:]
        self.data[class_id] = merged

    # ------------------------------------------------------------------
    def prototype(self, class_id: int) -> Optional[torch.Tensor]:
        return self.prototypes.get(class_id)

    def all_prototypes(self) -> Optional[torch.Tensor]:
        if not self.prototypes:
            return None
        keys = sorted(self.prototypes.keys())
        return torch.stack([self.prototypes[k] for k in keys], dim=0)

    def class_ids(self) -> List[int]:
        return sorted(self.prototypes.keys())

    def __len__(self) -> int:
        return len(self.prototypes)

    def __repr__(self) -> str:
        return (f"SlowMemory(classes={len(self.prototypes)}, "
                f"capacity_per_class={self.capacity_per_class})")