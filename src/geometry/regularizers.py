# mccl/src/mccl/geometry/regularizers.py
"""Régulariseurs géométriques par composante.

Objectifs :
- hyperbolique : éviter que les points collent au bord de la boule
  (problème classique de stabilité en RL hyperbolique) ;
- sphère : forcer les points à rester sur la sphère de rayon r ;
- euclidien : pénaliser la norme (léger).

Les régulariseurs sont appliqués dans l'espace ambient, pas sur les
coordonnées tangentes.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .manifolds import Euclidean, Hyperbolic, Sphere, ProductManifold


def reg_hyperbolic(x: Tensor, m: Hyperbolic) -> Tensor:
    """Pénalise les points proches du bord : ||x||^2 / max_norm^2."""
    n = torch.linalg.vector_norm(x, dim=-1)
    return (n ** 2 / (m.max_norm ** 2)).mean()


def reg_sphere(x: Tensor, m: Sphere) -> Tensor:
    """Force ||x|| = radius : moyenne de (||x|| - r)^2."""
    n = torch.linalg.vector_norm(x, dim=-1)
    return ((n - m.radius) ** 2).mean()


def reg_euclidean(x: Tensor, m: Euclidean) -> Tensor:
    """Pénalité L2 légère."""
    return (x ** 2).mean()


def combined_reg(x: Tensor, manifold) -> Tensor:
    """Somme des régulariseurs sur toutes les composantes.

    :param x: tenseur [..., dim] dans l'espace ambient du produit
    :param manifold: ProductManifold (ou Manifold unique)
    """
    if not isinstance(manifold, ProductManifold):
        if isinstance(manifold, Hyperbolic):
            return reg_hyperbolic(x, manifold)
        if isinstance(manifold, Sphere):
            return reg_sphere(x, manifold)
        if isinstance(manifold, Euclidean):
            return reg_euclidean(x, manifold)
        return torch.tensor(0.0, device=x.device)

    loss = torch.tensor(0.0, device=x.device)
    for sl, m in zip(manifold.slices, manifold.manifolds):
        xi = x[..., sl]
        if isinstance(m, Hyperbolic):
            loss = loss + reg_hyperbolic(xi, m)
        elif isinstance(m, Sphere):
            loss = loss + reg_sphere(xi, m)
        elif isinstance(m, Euclidean):
            loss = loss + reg_euclidean(xi, m)
    return loss


def weighted_combined_reg(
    x: Tensor,
    manifold,
    weights: dict,
) -> Tensor:
    """Version pondérée : weights = {'hyperbolic': λh, 'sphere': λs, ...}."""
    if not isinstance(manifold, ProductManifold):
        return combined_reg(x, manifold)

    loss = torch.tensor(0.0, device=x.device)
    for sl, m in zip(manifold.slices, manifold.manifolds):
        xi = x[..., sl]
        if isinstance(m, Hyperbolic):
            loss = loss + weights.get("hyperbolic", 1.0) * reg_hyperbolic(xi, m)
        elif isinstance(m, Sphere):
            loss = loss + weights.get("sphere", 1.0) * reg_sphere(xi, m)
        elif isinstance(m, Euclidean):
            loss = loss + weights.get("euclidean", 1.0) * reg_euclidean(xi, m)
    return loss