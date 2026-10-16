# mccl/src/mccl/evaluation/ood_metrics.py
"""Détection OOD : AUROC, FPR@95, et différents scores."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Aire sous la courbe ROC.

    Convention : un score élevé = in-distribution.
    """
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])

    # Tri décroissant
    order = np.argsort(-s)
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)

    # Trapèze
    return float(np.trapz(tpr, fpr))


def fpr_at_95_tpr(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """FPR au point où TPR = 95 %."""
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])

    order = np.argsort(-s)
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)

    idx = np.searchsorted(tpr, 0.95)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


# ---------------------------------------------------------------------------
# Scores OOD
# ---------------------------------------------------------------------------
@torch.no_grad()
def max_softmax_score(logits: torch.Tensor) -> np.ndarray:
    """Score max-softmax (MSP). Un score élevé = in-distribution."""
    p = torch.softmax(logits, dim=-1)
    return p.max(dim=-1).values.cpu().numpy()


@torch.no_grad()
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Energy score. Un score élevé = in-distribution.

    E(x) = T * logsumexp(logits / T)
    """
    return (temperature * torch.logsumexp(logits / temperature, dim=-1)).cpu().numpy()


@torch.no_grad()
def mahalanobis_score(
    features: torch.Tensor,
    means: torch.Tensor,
    cov_inv: torch.Tensor,
) -> np.ndarray:
    """Score de Mahalanobis (distance minimale aux centroïdes de classe).

    :param features: tenseur [N, D]
    :param means: tenseur [C, D], un centroïde par classe
    :param cov_inv: tenseur [D, D], inverse de la covariance partagée
    :return: array [N], score = -min_c d(x, mu_c)
    """
    diff = features.unsqueeze(1) - means.unsqueeze(0)   # [N, C, D]
    d2 = torch.einsum("ncd,de,nce->nc", diff, cov_inv, diff)
    min_d2 = d2.min(dim=-1).values
    return (-min_d2).cpu().numpy()


@torch.no_grad()
def entropy_score(logits: torch.Tensor) -> np.ndarray:
    """Entropie prédictive. Un score faible = in-distribution.

    Pour uniformiser avec les autres (score élevé = ID), on renvoie -H.
    """
    p = torch.softmax(logits, dim=-1)
    log_p = torch.log_softmax(logits, dim=-1)
    h = -(p * log_p).sum(dim=-1)
    return (-h).cpu().numpy()


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise un score en z-score."""
    mu = scores.mean()
    sigma = scores.std() + 1e-8
    return (scores - mu) / sigma


def combined_score(
    scores: dict,
    weights: Optional[dict] = None,
) -> np.ndarray:
    """Combine plusieurs scores OOD (moyenne pondérée après z-score)."""
    keys = list(scores.keys())
    if weights is None:
        weights = {k: 1.0 for k in keys}

    total = None
    for k in keys:
        z = normalize_scores(scores[k])
        contrib = weights.get(k, 1.0) * z
        total = contrib if total is None else total + contrib
    return total / sum(weights.values())