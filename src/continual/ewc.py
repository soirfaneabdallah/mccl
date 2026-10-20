# mccl/src/mccl/continual/ewc.py
"""Elastic Weight Consolidation (EWC) — baseline de comparaison.

Fisher diagonale empirique, estimée par échantillon et accumulée entre tâches
(variante « en ligne »). Prend en charge une tête de classification qui
s'étend : la pénalité ne porte que sur la partie existante lors de l'estimation.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from src.utils.logging import get_logger

log = get_logger(__name__)

LossFn = Callable[[Any, torch.Tensor], torch.Tensor]


def _overlap(p: torch.Tensor, ref: torch.Tensor) -> Tuple[slice, ...]:
    """Tranche de ``p`` couvrant la forme de ``ref`` (croissance en fin de dimension)."""
    if p.dim() != ref.dim() or any(r > s for r, s in zip(ref.shape, p.shape)):
        raise ValueError(f"Forme incompatible : paramètre {tuple(p.shape)}, référence {tuple(ref.shape)}")
    return tuple(slice(0, r) for r in ref.shape)


class EWC:
    """EWC (Kirkpatrick et al., 2017), Fisher diagonale empirique par échantillon.

    :param model: modèle dont les paramètres entraînables sont protégés
    :param lam: coefficient de pénalité (le terme est ``0.5 * lam * sum(F * (p - p*)**2)``)
    :param n_samples: nombre maximal d'**échantillons** utilisés pour estimer la Fisher
    :param gamma: décroissance de l'ancienne Fisher (1.0 = accumulation pure)
    """

    def __init__(
        self,
        model: nn.Module,
        lam: float = 1e4,
        n_samples: int = 512,
        gamma: float = 1.0,
    ) -> None:
        self.model = model
        self.lam = float(lam)
        self.n_samples = int(n_samples)
        self.gamma = float(gamma)

        self.fisher: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}
        self._n_estimated = 0
        self._warned_no_match = False

    def estimate(self, loss_fn: LossFn, batches: Iterable, device: str = "cpu") -> None:
        """Estime la Fisher sur au plus ``n_samples`` échantillons et la fusionne.

        :param loss_fn: ``(sortie_brute_du_modèle, y) -> perte scalaire``
        :param batches: itérable de ``(x, y)``
        :param device: device des données
        """
        was_training = self.model.training
        self.model.eval()
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        new_fisher = {n: torch.zeros_like(p) for n, p in params.items()}

        count = 0
        try:
            with torch.enable_grad():
                for batch in batches:
                    x, y = batch[0].to(device), batch[1].to(device)
                    for i in range(x.shape[0]):
                        self.model.zero_grad(set_to_none=True)
                        loss = loss_fn(self.model(x[i : i + 1]), y[i : i + 1])
                        loss.backward()
                        for n, p in params.items():
                            if p.grad is not None:
                                new_fisher[n] += p.grad.detach() ** 2
                        count += 1
                        if count >= self.n_samples:
                            break
                    if count >= self.n_samples:
                        break
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

        if count == 0:
            raise ValueError("EWC.estimate : aucun échantillon disponible")

        merged: Dict[str, torch.Tensor] = {}
        for n, f_new in new_fisher.items():
            f_new /= count
            old = self.fisher.get(n)
            if old is not None:
                padded = torch.zeros_like(f_new)
                padded[_overlap(f_new, old)] = old.to(f_new.device)
                f_new = self.gamma * padded + f_new
            merged[n] = f_new

        self.fisher = merged
        self.star = {n: params[n].detach().clone() for n in merged}
        self._n_estimated = count
        log.info("EWC : Fisher estimée sur %d échantillons (%d tenseurs)", count, len(merged))

    def penalty(self) -> torch.Tensor:
        """Terme de pénalité EWC (0 si aucune Fisher n'a été estimée)."""
        total = None
        matched = 0
        for name, p in self.model.named_parameters():
            fisher = self.fisher.get(name)
            if fisher is None:
                continue
            star = self.star[name].to(p.device)
            sl = _overlap(p, star)
            term = (fisher.to(p.device) * (p[sl] - star) ** 2).sum()
            total = term if total is None else total + term
            matched += 1

        if total is None:
            if self.fisher and not self._warned_no_match:
                log.warning("EWC : aucun nom de paramètre ne correspond (DDP/compile ?), pénalité nulle")
                self._warned_no_match = True
            return torch.zeros((), device=next(self.model.parameters()).device)
        return 0.5 * self.lam * total

    def stats(self) -> dict:
        return {
            "n_estimated": self._n_estimated,
            "lam": self.lam,
            "gamma": self.gamma,
            "n_params_tracked": len(self.fisher),
        }