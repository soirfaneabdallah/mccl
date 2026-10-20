# mccl/src/mccl/continual/surprise.py
"""Score de surprise à partir de l'erreur de prédiction JEPA.

Le score est normalisé par une moyenne/variance glissante, ce qui
permet de comparer des erreurs de magnitudes différentes au fil du temps.
"""
from __future__ import annotations

from typing import Optional

import torch


class SurpriseScorer:
    """Calcule un score de surprise normalisé.

    - Entrée : erreur de prédiction JEPA (ou toute autre mesure d'erreur).
    - Sortie : z-score par rapport à une estimation en ligne (EMA).
    """

    def __init__(
        self,
        alpha: float = 0.99,
        min_var: float = 1e-6,
    ) -> None:
        self.alpha = float(alpha)
        self.min_var = float(min_var)

        self.mean: Optional[float] = None
        self.var: Optional[float] = None
        self._initialized = False

    # ------------------------------------------------------------------
    def update(self, err: torch.Tensor) -> torch.Tensor:
        """Met à jour les statistiques et renvoie les z-scores.

        :param err: tenseur [B] d'erreurs (JEPA, MSE, etc.)
        :return: tenseur [B] de scores normalisés
        """
        err = err.detach().float().flatten()
        if err.numel() == 0:
            return err

        batch_mean = err.mean().item()
        batch_var = err.var(unbiased=False).item() if err.numel() > 1 else 0.0
        batch_var = max(batch_var, self.min_var)

        if not self._initialized:
            self.mean = batch_mean
            self.var = batch_var
            self._initialized = True
        else:
            a = self.alpha
            self.mean = a * self.mean + (1 - a) * batch_mean
            self.var = a * self.var + (1 - a) * batch_var

        return (err - self.mean) / (self.var ** 0.5 + 1e-8)

    # ------------------------------------------------------------------
    def threshold(self, scores: torch.Tensor, quantile: float = 0.9) -> float:
        """Renvoie le seuil correspondant à un quantile des scores."""
        if scores.numel() == 0:
            return 0.0
        return float(torch.quantile(scores, quantile).item())

    def stats(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "initialized": self._initialized,
        }

    def reset(self) -> None:
        self.mean = None
        self.var = None
        self._initialized = False