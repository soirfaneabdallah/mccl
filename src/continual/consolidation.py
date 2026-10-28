# mccl/src/mccl/continual/consolidation.py
"""Consolidation : transfert des éléments surprenants vers la mémoire lente.

Le déclenchement repose sur un seuil de surprise calculé sur une FENÊTRE
GLISSANTE de l'historique (quantile glissant), et non sur le batch courant :
un batch entièrement familier ne déclenche donc (presque) aucune consolidation,
alors qu'un changement de distribution (nouvelle tâche) en déclenche beaucoup.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from .memory import SlowMemory

logger = logging.getLogger(__name__)


class Consolidator:
    """Décide quels éléments consolider, et effectue le transfert.

    Fonctionnement :
    1. Reçoit un batch de features et un score de surprise.
    2. Compare chaque score au quantile de la fenêtre glissante d'historique.
    3. Transfère les éléments retenus vers SlowMemory, ou les met en attente
       dans un buffer borné (FIFO) si aucune SlowMemory n'est fournie.

    :param threshold_quantile: quantile de l'historique servant de seuil (0, 1)
    :param window_size: taille de la fenêtre glissante des scores passés
    :param warmup: nb minimal de scores en historique avant de s'y fier ; en
        dessous, le seuil est calculé sur historique + batch courant
    :param max_per_batch: plafond de sélection par appel (top-k par surprise) ;
        évite qu'un pic de surprise ne consolide tout un batch d'un coup
    :param max_buffer: taille maximale du buffer d'attente (FIFO)
    :param store_device: device des features transmises à SlowMemory / buffer
    :param min_buffer: alias DÉPRÉCIÉ de ``warmup`` (compatibilité)
    """

    def __init__(
        self,
        threshold_quantile: float = 0.9,
        window_size: int = 2048,
        warmup: int = 32,
        max_per_batch: Optional[int] = None,
        max_buffer: int = 4096,
        store_device: str = "cpu",
        min_buffer: Optional[int] = None,
    ) -> None:
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile doit être dans ]0, 1[")
        if window_size < 1 or max_buffer < 1:
            raise ValueError("window_size et max_buffer doivent être >= 1")
        if min_buffer is not None:
            logger.warning("min_buffer est déprécié : utiliser warmup.")
            warmup = min_buffer

        self.threshold_quantile = threshold_quantile
        self.window_size = window_size
        self.warmup = warmup
        self.max_per_batch = max_per_batch
        self.max_buffer = max_buffer
        self.store_device = torch.device(store_device)

        # Fenêtre glissante (buffer circulaire), créée au 1er appel sur le
        # device des scores pour éviter les transferts GPU->CPU à chaque pas.
        self._win: Optional[torch.Tensor] = None
        self._win_pos = 0
        self._win_count = 0

        # Buffer d'attente borné (FIFO) : listes de chunks + taille totale
        self._buf_feats: Deque[torch.Tensor] = deque()
        self._buf_labels: Deque[torch.Tensor] = deque()
        self._buf_size = 0

        # Compteurs
        self.n_seen = 0
        self.n_selected = 0       # éléments retenus par le seuil
        self.n_consolidated = 0   # éléments réellement envoyés à SlowMemory
        self.n_dropped = 0        # éléments évincés du buffer (FIFO plein)
        self.last_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Fenêtre glissante
    # ------------------------------------------------------------------
    def _push_window(self, scores: torch.Tensor) -> None:
        """Ajoute des scores à la fenêtre circulaire."""
        if self._win is None or self._win.device != scores.device:
            old = self._win
            self._win = torch.zeros(
                self.window_size, dtype=torch.float32, device=scores.device
            )
            if old is not None:
                self._win[: old.numel()] = old.to(scores.device)

        scores = scores[-self.window_size :]
        n = scores.numel()
        end = self._win_pos + n
        if end <= self.window_size:
            self._win[self._win_pos : end] = scores
        else:
            k = self.window_size - self._win_pos
            self._win[self._win_pos :] = scores[:k]
            self._win[: n - k] = scores[k:]
        self._win_pos = end % self.window_size
        self._win_count = min(self._win_count + n, self.window_size)

    def _threshold(self, scores: torch.Tensor) -> torch.Tensor:
        """Quantile de l'historique ; pendant le warmup, historique + batch."""
        if self._win is not None and self._win_count > 0:
            history = self._win[: self._win_count].to(scores.device)
        else:
            history = scores.new_empty(0)
        ref = history if self._win_count >= self.warmup else torch.cat([history, scores])
        return torch.quantile(ref, self.threshold_quantile)

    # ------------------------------------------------------------------
    # Sélection + consolidation
    # ------------------------------------------------------------------
    @torch.no_grad()
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
        :return: nombre d'éléments sélectionnés
        """
        if surprise.numel() == 0 or feats.numel() == 0:
            return 0
        if not (surprise.shape[0] == feats.shape[0] == labels.shape[0]):
            raise ValueError(
                f"Tailles incohérentes : surprise={tuple(surprise.shape)}, "
                f"feats={tuple(feats.shape)}, labels={tuple(labels.shape)}"
            )

        surprise = surprise.detach().flatten().float()
        feats = feats.detach()
        labels = labels.detach().flatten()

        # Un seul NaN fausserait tout le quantile : on écarte les non-finis.
        finite = torch.isfinite(surprise)
        if not bool(finite.all()):
            logger.warning(
                "%d score(s) de surprise non fini(s) ignoré(s).",
                int((~finite).sum()),
            )
            surprise, feats, labels = surprise[finite], feats[finite], labels[finite]
            if surprise.numel() == 0:
                return 0

        self.n_seen += surprise.numel()

        # Seuil calculé sur l'HISTORIQUE (avant d'y ajouter le batch courant).
        thr = self._threshold(surprise)
        self.last_threshold = float(thr)
        self._push_window(surprise)

        # Inégalité stricte : si tous les scores sont égaux (ex. 0), rien
        # n'est sélectionné au lieu de tout sélectionner.
        idx = (surprise > thr).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return 0

        if self.max_per_batch is not None and idx.numel() > self.max_per_batch:
            top = surprise[idx].topk(self.max_per_batch).indices
            idx = idx[top]

        sel_feats = feats[idx].to(self.store_device)
        sel_labels = labels[idx].to(self.store_device)
        n = int(idx.numel())
        self.n_selected += n

        if slow_memory is not None:
            for c in sel_labels.unique().tolist():
                slow_memory.update(int(c), sel_feats[sel_labels == c])
            self.n_consolidated += n
        else:
            self._buffer_add(sel_feats, sel_labels)

        return n

    # ------------------------------------------------------------------
    # Buffer d'attente borné
    # ------------------------------------------------------------------
    def _buffer_add(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        self._buf_feats.append(feats)
        self._buf_labels.append(labels)
        self._buf_size += feats.size(0)

        # Éviction FIFO : on retire les plus anciens au-delà de max_buffer.
        while self._buf_size > self.max_buffer:
            excess = self._buf_size - self.max_buffer
            oldest = self._buf_feats[0].size(0)
            if oldest <= excess:
                self._buf_feats.popleft()
                self._buf_labels.popleft()
                self._buf_size -= oldest
                self.n_dropped += oldest
            else:
                self._buf_feats[0] = self._buf_feats[0][excess:]
                self._buf_labels[0] = self._buf_labels[0][excess:]
                self._buf_size -= excess
                self.n_dropped += excess

    def drain(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Vide le buffer et renvoie (features [N, D], labels [N]).

        NB : les labels sont désormais renvoyés sous forme de tenseur (et non
        de ``list[int]``) ; utiliser ``.tolist()`` si une liste est requise.
        """
        if not self._buf_feats:
            return None, None
        feats = torch.cat(list(self._buf_feats), dim=0)
        labels = torch.cat(list(self._buf_labels), dim=0)
        self._buf_feats.clear()
        self._buf_labels.clear()
        self._buf_size = 0
        # Le buffer vidé a été consommé par l'appelant : on le comptabilise.
        self.n_consolidated += feats.size(0)
        return feats, labels

    # ------------------------------------------------------------------
    # Introspection / persistance
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "n_seen": self.n_seen,
            "n_selected": self.n_selected,
            "n_consolidated": self.n_consolidated,
            "n_dropped": self.n_dropped,
            "buffer_size": self._buf_size,
            "window_fill": self._win_count,
            "last_threshold": self.last_threshold,
            "selection_rate": self.n_selected / self.n_seen if self.n_seen else 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """État sérialisable (checkpoint d'apprentissage continu)."""
        return {
            "window": None if self._win is None else self._win.cpu(),
            "win_pos": self._win_pos,
            "win_count": self._win_count,
            "buf_feats": [f.cpu() for f in self._buf_feats],
            "buf_labels": [l.cpu() for l in self._buf_labels],
            "counters": {
                "n_seen": self.n_seen,
                "n_selected": self.n_selected,
                "n_consolidated": self.n_consolidated,
                "n_dropped": self.n_dropped,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._win = state["window"]
        self._win_pos = state["win_pos"]
        self._win_count = state["win_count"]
        self._buf_feats = deque(state["buf_feats"])
        self._buf_labels = deque(state["buf_labels"])
        self._buf_size = sum(f.size(0) for f in self._buf_feats)
        c = state["counters"]
        self.n_seen = c["n_seen"]
        self.n_selected = c["n_selected"]
        self.n_consolidated = c["n_consolidated"]
        self.n_dropped = c["n_dropped"]