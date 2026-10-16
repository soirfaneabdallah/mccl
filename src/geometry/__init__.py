# mccl/src/mccl/geometry/__init__.py
"""Variétés riemanniennes, régulariseurs et diagnostics pour MCCL."""

from .manifolds import (
    Manifold,
    Euclidean,
    Sphere,
    Hyperbolic,
    ProductManifold,
    build_manifold,
)
from .regularizers import (
    reg_hyperbolic,
    reg_sphere,
    reg_euclidean,
    combined_reg,
)
from .diagnostics import (
    delta_hyperbolicity,
    curvature_probe,
    distortion,
    tangent_alignment,
)

__all__ = [
    "Manifold", "Euclidean", "Sphere", "Hyperbolic", "ProductManifold",
    "build_manifold",
    "reg_hyperbolic", "reg_sphere", "reg_euclidean", "combined_reg",
    "delta_hyperbolicity", "curvature_probe", "distortion", "tangent_alignment",
]