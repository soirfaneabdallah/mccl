# mccl/src/mccl/training/losses.py
"""Pertes pour MCCL : JEPA, classification, régularisation géométrique."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# JEPA
# ---------------------------------------------------------------------------
def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    manifold=None,
) -> torch.Tensor:
    """Perte JEPA simple : MSE dans l'espace tangent (T_0).

    Si `manifold` est fourni et que pred/target sont sur la variété,
    la perte devient d_M². Sinon, MSE classique.

    :param pred: [B, D] prédiction (dans T_0 ou sur la variété)
    :param target: [B, D] cible (idem)
    :param manifold: optionnel, ProductManifold pour dist_sq
    """
    if manifold is not None:
        return manifold.dist_sq(pred, target).mean()
    return F.mse_loss(pred, target)


def jepa_multi_step(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    manifold=None,
    discount: float = 1.0,
) -> torch.Tensor:
    """Perte JEPA multi-pas avec facteur d'actualisation.

    :param preds: liste de prédictions [B, D] pour chaque pas
    :param targets: liste de cibles [B, D] pour chaque pas
    :param manifold: optionnel, ProductManifold
    :param discount: facteur par pas (1.0 = pas de discount)
    """
    if len(preds) != len(targets):
        raise ValueError("preds et targets doivent avoir la même longueur")

    total = torch.tensor(0.0, device=preds[0].device)
    weight_sum = 0.0
    for i, (p, t) in enumerate(zip(preds, targets)):
        w = discount ** i
        total = total + w * jepa_loss(p, t, manifold)
        weight_sum += w
    return total / max(weight_sum, 1e-8)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Entropie croisée standard.

    :param logits: [B, C]
    :param y: [B] entiers dans [0, C)
    """
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


def prototype_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Perte pour classifieur à prototypes.

    Identique à cross_entropy, mais la température est déjà dans les logits.
    Ajouté pour clarté sémantique.
    """
    return F.cross_entropy(logits / temperature, y)


# ---------------------------------------------------------------------------
# Régularisation géométrique
# ---------------------------------------------------------------------------
def reg_loss(
    feats: torch.Tensor,
    manifold,
    cfg=None,
) -> torch.Tensor:
    """Régulariseur géométrique combiné, pondéré par cfg.

    :param feats: tenseur [B, N, D] ou [B, D] sur la variété
    :param manifold: ProductManifold
    :param cfg: objet avec lambda_h, lambda_s, lambda_e (RegCfg)
    """
    from geometry.regularizers import combined_reg

    reg = combined_reg(feats, manifold)

    if cfg is None:
        return reg

    # Pondération par type de variété
    weights = {
        "hyperbolic": getattr(cfg, "lambda_h", 1.0),
        "sphere": getattr(cfg, "lambda_s", 1.0),
        "euclidean": getattr(cfg, "lambda_e", 1.0),
    }
    # Poids scalaire global (moyenne pondérée simple)
    # On applique directement combined_reg déjà sommé, puis on module.
    scale = sum(weights.values()) / 3.0
    return reg * scale


# ---------------------------------------------------------------------------
# Perte totale
# ---------------------------------------------------------------------------
def total_loss(
    out,
    y: Optional[torch.Tensor] = None,
    manifold=None,
    reg_cfg=None,
    lambda_jepa: float = 1.0,
    lambda_cls: float = 1.0,
    lambda_reg: float = 1.0e-3,
) -> torch.Tensor:
    """Perte totale = JEPA + classification + régularisation géométrique.

    :param out: JepaOutput
    :param y: labels pour la classification (optionnel)
    :param manifold: ProductManifold
    :param reg_cfg: config de régularisation
    """
    device = out.tokens.device
    loss = torch.tensor(0.0, device=device)

    # Classification
    if y is not None and out.logits is not None:
        loss = loss + lambda_cls * classification_loss(out.logits, y)

    # JEPA
    if out.pred is not None and out.target is not None:
        loss = loss + lambda_jepa * jepa_loss(out.pred, out.target, manifold)

    # Régularisation
    if manifold is not None:
        loss = loss + lambda_reg * reg_loss(out.tokens, manifold, reg_cfg)

    return loss


# ---------------------------------------------------------------------------
# Perte contrastive (optionnelle, pour ablations)
# ---------------------------------------------------------------------------
def contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Perte contrastive supervisée (SupCon).

    :param z: [B, D] embeddings (à normaliser)
    :param y: [B] labels
    :param temperature: température
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temperature              # [B, B]
    B = z.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float("-inf"))

    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & mask

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=-1, keepdim=True)
    log_prob = sim - torch.log(denom.clamp_min(1e-12))

    pos_counts = pos_mask.sum(dim=-1).clamp_min(1)
    loss_per = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_counts
    return loss_per.mean()