# mccl/src/mccl/geometry/manifolds.py
"""Variétés riemanniennes : euclidienne, sphère, hyperbolique (Poincaré),
et produit riemannien.

Conventions :
- boule de Poincaré de courbure -c, avec c > 0 paramétrable ;
- sphère unité de rayon r, plongée dans R^dim ;
- produit riemannien avec distance d^2 = sum_i d_i^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

__all__ = [
    "Manifold",
    "Euclidean",
    "Sphere",
    "Hyperbolic",
    "ProductManifold",
    "build_manifold",
]


class Manifold:
    """Interface minimale pour les variétés utilisées dans MCCL."""

    dim: int

    def expmap0(self, v: Tensor) -> Tensor:
        raise NotImplementedError

    def logmap0(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        return self.dist(x, y) ** 2

    def proj(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Euclidien
# ---------------------------------------------------------------------------
@dataclass
class Euclidean(Manifold):
    """Espace euclidien R^dim."""

    dim: int

    def expmap0(self, v: Tensor) -> Tensor:
        return v

    def logmap0(self, y: Tensor) -> Tensor:
        return y

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.linalg.vector_norm(x - y, dim=-1)

    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        return ((x - y) ** 2).sum(dim=-1)

    def proj(self, x: Tensor) -> Tensor:
        return x


# ---------------------------------------------------------------------------
# Sphère
# ---------------------------------------------------------------------------
@dataclass
class Sphere(Manifold):
    """Sphère unité S^{dim-1} plongée dans R^dim.

    Point base : e0 = (1, 0, ..., 0).
    Espace tangent en e0 : {v : v_0 = 0}.
    """

    dim: int
    radius: float = 1.0

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("Sphere.dim doit être >= 2.")
        if self.radius <= 0:
            raise ValueError("Sphere.radius doit être > 0.")

    def _p0(self, x: Tensor) -> Tensor:
        p = torch.zeros_like(x)
        p[..., 0] = 1.0
        return p

    def _tangent_proj(self, v: Tensor) -> Tensor:
        v = v.clone()
        v[..., 0] = 0.0
        return v

    def expmap0(self, v: Tensor) -> Tensor:
        v = self._tangent_proj(v)
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        sin_over_norm = torch.where(
            norm > 0,
            torch.sin(norm) / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        p0 = self._p0(v)
        y = torch.cos(norm) * p0 + sin_over_norm * v
        return y * self.radius

    def logmap0(self, y: Tensor) -> Tensor:
        y = y / self.radius
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)

        p0 = self._p0(y)
        dot = (y * p0).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot)

        tangent = y - dot * p0
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)

        scale = torch.where(
            tangent_norm > 0,
            theta / tangent_norm.clamp_min(1e-12),
            torch.ones_like(tangent_norm),
        )
        return scale * tangent

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
        dot = (x * y).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(dot) * self.radius

    def proj(self, x: Tensor) -> Tensor:
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
        return self.radius * x / n


# ---------------------------------------------------------------------------
# Hyperbolique (boule de Poincaré)
# ---------------------------------------------------------------------------
@dataclass
class Hyperbolic(Manifold):
    """Boule de Poincaré de dimension `dim`, courbure -c avec c > 0."""

    dim: int
    curvature: float = 1.0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("Hyperbolic.dim doit être >= 1.")
        if self.curvature <= 0:
            raise ValueError("Hyperbolic.curvature doit être > 0.")
        if not (0 < self.eps < 1):
            raise ValueError("Hyperbolic.eps doit être dans (0, 1).")

    @property
    def sqrt_c(self) -> float:
        return math.sqrt(self.curvature)

    @property
    def max_norm(self) -> float:
        return (1.0 - self.eps) / self.sqrt_c

    def expmap0(self, v: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        scale = torch.where(
            norm > 0,
            torch.tanh(self.sqrt_c * norm / 2.0)
            / (self.sqrt_c * norm.clamp_min(1e-12)),
            torch.ones_like(norm),
        )
        return self.proj(scale * v)

    def logmap0(self, y: Tensor) -> Tensor:
        y = self.proj(y)
        norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)

        x = (self.sqrt_c * norm).clamp(max=1.0 - 1e-7)
        artanh = 0.5 * torch.log((1.0 + x) / (1.0 - x))

        scale = torch.where(
            norm > 0,
            (2.0 / self.sqrt_c) * artanh / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        return scale * y

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        c = self.curvature
        x = self.proj(x)
        y = self.proj(y)

        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        diff2 = ((x - y) ** 2).sum(dim=-1)

        denom = (1.0 - c * x2).clamp_min(1e-7) * (1.0 - c * y2).clamp_min(1e-7)
        arg = (1.0 + 2.0 * c * diff2 / denom).clamp_min(1.0)

        return torch.acosh(arg) / self.sqrt_c

    def proj(self, x: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        max_norm = self.max_norm
        scale = torch.where(
            norm > max_norm,
            max_norm / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        return x * scale


# ---------------------------------------------------------------------------
# Produit riemannien
# ---------------------------------------------------------------------------
@dataclass
class ProductManifold(Manifold):
    """Produit riemannien de plusieurs variétés.

    La distance produit est d(x, y) = sqrt(sum_i d_i(x_i, y_i)^2).
    """

    manifolds: Sequence[Manifold]

    def __post_init__(self) -> None:
        if len(self.manifolds) == 0:
            raise ValueError("ProductManifold nécessite au moins une variété.")
        self.dims = [m.dim for m in self.manifolds]
        self.dim = sum(self.dims)
        self.slices = []
        start = 0
        for d in self.dims:
            self.slices.append(slice(start, start + d))
            start += d

    def expmap0(self, v: Tensor) -> Tensor:
        return torch.cat(
            [m.expmap0(v[..., sl]) for sl, m in zip(self.slices, self.manifolds)],
            dim=-1,
        )

    def logmap0(self, y: Tensor) -> Tensor:
        return torch.cat(
            [m.logmap0(y[..., sl]) for sl, m in zip(self.slices, self.manifolds)],
            dim=-1,
        )

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        ds = torch.stack(
            [m.dist(x[..., sl], y[..., sl])
             for sl, m in zip(self.slices, self.manifolds)],
            dim=-1,
        )
        return torch.sqrt((ds ** 2).sum(dim=-1))

    def dist_sq(self, x: Tensor, y: Tensor) -> Tensor:
        ds = torch.stack(
            [m.dist(x[..., sl], y[..., sl])
             for sl, m in zip(self.slices, self.manifolds)],
            dim=-1,
        )
        return (ds ** 2).sum(dim=-1)

    def proj(self, x: Tensor) -> Tensor:
        return torch.cat(
            [m.proj(x[..., sl]) for sl, m in zip(self.slices, self.manifolds)],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Construction depuis une config
# ---------------------------------------------------------------------------
def build_manifold(cfg) -> ProductManifold:
    """Construit un ProductManifold depuis une config `ManifoldCfg`.

    :param cfg: objet avec un attribut `.components` (liste de dicts
        {'type': ..., 'dim': ..., 'curvature': optionnel})
    """
    comps = []
    for c in cfg.components:
        t = c["type"]
        if t == "euclidean":
            comps.append(Euclidean(c["dim"]))
        elif t == "sphere":
            comps.append(Sphere(c["dim"]))
        elif t == "hyperbolic":
            comps.append(Hyperbolic(c["dim"], c.get("curvature", 1.0)))
        else:
            raise ValueError(f"Type de variété inconnu : {t}")
    return ProductManifold(comps)