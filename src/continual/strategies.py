# mccl/src/mccl/continual/strategies.py
"""Stratégies d'apprentissage continu avec interface complète.

Chaque stratégie expose :
- begin_task(task) : préparation avant entraînement
- observe(batch, out) : observation d'un batch
- end_task(task, loader=None) : finalisation, éventuellement estimation Fisher
- sample_replay(n) : échantillon de rejeu (x, y) ou None
- penalty() : terme de régularisation à ajouter à la perte (0 par défaut)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logging import get_logger

from .memory import FastMemory, SlowMemory
from .surprise import SurpriseScorer
from .consolidation import Consolidator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Interface de base
# ---------------------------------------------------------------------------
class Strategy:
    """Interface commune pour toutes les stratégies."""

    name: str = "base"

    def begin_task(self, task) -> None:
        """Prépare la stratégie pour une nouvelle tâche."""
        pass

    def observe(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        out=None,
    ) -> None:
        """Observe un batch d'entraînement (pour peupler les mémoires)."""
        pass

    def end_task(self, task, train_loader=None) -> None:
        """Finalise la tâche, éventuellement avec accès aux données."""
        pass

    def sample_replay(
        self, n: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Renvoie un batch de rejeu ou None."""
        return None

    def penalty(self) -> torch.Tensor:
        """Terme de régularisation à ajouter à la perte (0 par défaut)."""
        return torch.tensor(0.0)

    def state(self) -> Dict[str, Any]:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Finetune : baseline basse
# ---------------------------------------------------------------------------
class Finetune(Strategy):
    """Aucune protection. Oubli catastrophique attendu."""

    name = "finetune"


# ---------------------------------------------------------------------------
# Joint : baseline haute (utilise toutes les données vues)
# ---------------------------------------------------------------------------
class Joint(Strategy):
    """Accès à toutes les classes vues à chaque tâche.

    Suppose que l'entraîneur construit le loader sur `task.seen_classes`
    au lieu de `task.classes`.
    """

    name = "joint"


# ---------------------------------------------------------------------------
# Replay simple
# ---------------------------------------------------------------------------
class Replay(Strategy):
    """Rejeu uniforme depuis un réservoir borné."""

    name = "replay"

    def __init__(
        self,
        capacity: int = 512,
        replay_ratio: float = 1.0,
        strategy: str = "reservoir",
    ) -> None:
        self.memory = FastMemory(capacity=capacity, strategy=strategy)
        self.replay_ratio = float(replay_ratio)

    def observe(self, batch, out=None) -> None:
        x, y = batch
        self.memory.add(x.detach().cpu(), y.detach().cpu())

    def sample_replay(self, n: int):
        x, y, _ = self.memory.sample(n)
        if x is None:
            return None
        return x, y

    def state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "memory_size": len(self.memory),
            "class_counts": self.memory.class_counts(),
        }


# ---------------------------------------------------------------------------
# EWC : régularisation par Fisher diagonale
# ---------------------------------------------------------------------------
class EWCMethod(Strategy):
    """EWC (Kirkpatrick et al., 2017).

    Nécessite un entraînement en deux temps :
    1. entraîner normalement (avec penalty() appliquée si déjà estimée) ;
    2. appeler end_task(task, train_loader) pour estimer la Fisher
       sur les données de la tâche, puis fusionner.
    """

    name = "ewc"

    def __init__(
        self,
        lam: float = 1e4,
        n_samples: int = 512,
        gamma: float = 1.0,
    ) -> None:
        self.lam = float(lam)
        self.n_samples = int(n_samples)
        self.gamma = float(gamma)
        self.ewc = None      # instance de EWC, créée à la première end_task
        self._model = None

    def attach_model(self, model: nn.Module) -> None:
        """Associe le modèle. Doit être appelé après l'instanciation."""
        from .ewc import EWC
        self._model = model
        self.ewc = EWC(model=model, lam=self.lam,
                       n_samples=self.n_samples, gamma=self.gamma)

    def penalty(self) -> torch.Tensor:
        if self.ewc is None:
            return torch.tensor(0.0)
        return self.ewc.penalty()

    def end_task(self, task, train_loader=None) -> None:
        """Estime la Fisher sur les données de la tâche."""
        if self.ewc is None:
            raise RuntimeError("EWCMethod.attach_model(model) doit être appelé avant end_task")
        if train_loader is None:
            log.warning("EWCMethod.end_task : pas de train_loader, Fisher non estimée")
            return

        # --- CORRECTION ICI ---
        def loss_fn(logits, y):
            # logits peut être un JepaOutput ou un tenseur
            logits = logits.logits if hasattr(logits, "logits") else logits
            
            # SÉCURITÉ : Si le modèle n'a pas assez de sorties pour le label y,
            # on ignore ce batch (évite l'erreur IndexError).
            if logits.shape[-1] <= y.max():
                log.warning(f"EWC : Label {y.max().item()} hors limites (sortie: {logits.shape[-1]}). Batch ignoré.")
                return torch.tensor(0.0, requires_grad=True, device=logits.device)
                
            return torch.nn.functional.cross_entropy(logits, y)
        # ----------------------

        self.ewc.estimate(loss_fn, train_loader, device=str(next(self._model.parameters()).device))
        log.info("EWC : Fisher mise à jour sur la tâche %d", task.index)

    def state(self) -> Dict[str, Any]:
        s = {"name": self.name, "lam": self.lam, "gamma": self.gamma}
        if self.ewc is not None:
            s.update(self.ewc.stats())
        return s


# ---------------------------------------------------------------------------
# DualMemory : fast + slow + surprise + consolidation
# ---------------------------------------------------------------------------
class DualMemory(Strategy):
    """Mémoire duale : rejeu rapide + consolidation pilotée par la surprise."""

    name = "dual_memory"

    def __init__(
        self,
        fast_cfg: dict = None,
        slow_cfg: dict = None,
        surprise_cfg: dict = None,
        consolidation_cfg: dict = None,
    ) -> None:
        fast_cfg = fast_cfg or {}
        slow_cfg = slow_cfg or {}
        surprise_cfg = surprise_cfg or {}
        consolidation_cfg = consolidation_cfg or {}

        self.fast = FastMemory(
            capacity=fast_cfg.get("capacity", 256),
            strategy=fast_cfg.get("strategy", "reservoir"),
        )
        self.slow = SlowMemory(
            capacity_per_class=slow_cfg.get("capacity_per_class", 32),
        )
        self.surprise = SurpriseScorer(
            alpha=surprise_cfg.get("alpha", 0.99),
        )
        self.consolidator = Consolidator(
            threshold_quantile=consolidation_cfg.get("threshold_quantile", 0.9),
        )

        self.n_consolidated = 0
        self.task_idx = 0

    def begin_task(self, task) -> None:
        self.task_idx = getattr(task, "index", 0)
        log.info("[DualMemory] début tâche %d | classes=%s",
                 self.task_idx, getattr(task, "classes", []))

    def observe(self, batch, out=None) -> None:
        x, y = batch
        self.fast.add(x.detach().cpu(), y.detach().cpu())

        # Surprise + consolidation si on a une sortie JEPA
        if out is not None and getattr(out, "pred", None) is not None \
                and getattr(out, "target", None) is not None:
            err = ((out.pred - out.target) ** 2).mean(dim=-1)
            scores = self.surprise.update(err)

            feats = getattr(out, "global_embed", None)
            if feats is not None:
                n = self.consolidator.maybe_store(
                    surprise=scores,
                    feats=feats.detach().cpu(),
                    labels=y.detach().cpu(),
                    slow_memory=self.slow,
                )
                self.n_consolidated += n

    def sample_replay(self, n: int):
        x, y, _ = self.fast.sample(n)
        if x is None:
            return None
        return x, y

    def state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task_idx,
            "n_fast": len(self.fast),
            "n_slow": len(self.slow),
            "n_consolidated": self.n_consolidated,
            "surprise": self.surprise.stats(),
            "consolidator": self.consolidator.stats(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_strategy(name: str, **kwargs) -> Strategy:
    """Construit une stratégie par son nom."""
    name = name.lower()
    if name == "finetune":
        return Finetune()
    if name == "joint":
        return Joint()
    if name == "replay":
        return Replay(**kwargs)
    if name == "ewc":
        return EWCMethod(**kwargs)
    if name == "dual_memory":
        return DualMemory(**kwargs)
    raise ValueError(f"Stratégie inconnue : {name}")