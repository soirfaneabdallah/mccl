# mccl/src/mccl/geometry/diagnostics.py — ajouts
"""Diagnostics géométriques : δ-hyperbolicité, courbure, distorsion."""

import torch
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 1. δ-hyperbolicité (Gromov)
# ---------------------------------------------------------------------------
def delta_hyperbolicity(
    points: torch.Tensor,
    n_quadruples: int = 512,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Estimation Monte-Carlo de la δ-hyperbolicité.

    Pour un quadruple (x,y,z,w), on calcule les trois sommes de distances
    opposées. L'excès du plus grand sur le second est un proxy de 2δ.

    :param points: [N, D] points dans l'espace ambient
    :param n_quadruples: nombre de quadruples échantillonnés
    :return: δ estimé (petit = hyperbolique, grand = euclidien)
    """
    N = points.shape[0]
    if N < 4:
        return 0.0

    idx = torch.randint(0, N, (n_quadruples, 4), generator=generator)
    x, y, z, w = (points[idx[:, i]] for i in range(4))

    def d(a, b):
        return torch.linalg.vector_norm(a - b, dim=-1)

    s1 = d(x, y) + d(z, w)
    s2 = d(x, z) + d(y, w)
    s3 = d(x, w) + d(y, z)

    sums = torch.stack([s1, s2, s3], dim=-1)              # [Q, 3]
    sums_sorted, _ = torch.sort(sums, dim=-1, descending=True)
    delta = (sums_sorted[..., 0] - sums_sorted[..., 1]) / 2.0
    return float(delta.mean().item())


# ---------------------------------------------------------------------------
# 2. Courbure estimée (multiple méthodes)
# ---------------------------------------------------------------------------
def curvature_probe(points: torch.Tensor) -> float:
    """Proxy simple : norme moyenne des points dans l'espace ambient.

    Dans la boule de Poincaré, une norme proche de 1 indique une structure
    plus « hiérarchique ».
    """
    n = torch.linalg.vector_norm(points, dim=-1)
    return float(n.mean().item())


def curvature_from_distances(
    points: torch.Tensor,
    n_triples: int = 512,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Estimation de courbure par la formule de Toponogov.

    Pour trois points formant un triangle, on compare la somme des angles
    à π :
    - en euclidien : π ;
    - en hyperbolique : < π ;
    - en sphérique : > π.

    :return: déviation moyenne à π (négatif = hyperbolique)
    """
    N = points.shape[0]
    if N < 3:
        return 0.0

    idx = torch.randint(0, N, (n_triples, 3), generator=generator)
    a, b, c = (points[idx[:, i]] for i in range(3))

    def angle(u, v, w):
        vu = u - v
        vw = w - v
        dot = (vu * vw).sum(dim=-1)
        nu = torch.linalg.vector_norm(vu, dim=-1).clamp_min(1e-12)
        nw = torch.linalg.vector_norm(vw, dim=-1).clamp_min(1e-12)
        return torch.acos((dot / (nu * nw)).clamp(-1 + 1e-7, 1 - 1e-7))

    sum_angles = angle(b, a, c) + angle(a, b, c) + angle(a, c, b)
    deviation = (sum_angles - torch.pi).mean()
    return float(deviation.item())


def local_curvature_estimate(
    points: torch.Tensor,
    k: int = 10,
    n_samples: int = 256,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """Estimation locale de courbure par k-NN.

    Pour chaque point, on regarde ses k plus proches voisins. La croissance
    du volume de la boule est comparée à la croissance euclidienne.

    - Volume croît plus vite que r^D → courbure négative (hyperbolique).
    - Volume croît moins vite → courbure positive (sphérique).

    :return: dict avec 'mean_curvature', 'std_curvature', 'median_curvature'
    """
    N = points.shape[0]
    if N < k + 1:
        return {"mean_curvature": 0.0, "std_curvature": 0.0, "median_curvature": 0.0}

    idx = torch.randint(0, N, (n_samples,), generator=generator)
    samples = points[idx]

    dists = torch.cdist(samples, points)                  # [S, N]
    # Exclure la distance 0 (le point lui-même)
    dists, _ = torch.sort(dists, dim=-1)
    dists = dists[:, 1:k+1]                               # [S, k]

    # Rayon moyen et volume approximé par r^k
    mean_r = dists[:, -1].clamp_min(1e-8)                 # rayon à k
    volume_proxy = mean_r ** k                            # ~ V(B_r) à facteur près

    # En euclidien, log(V) / log(r) ≈ D. Déviation → courbure.
    log_v = torch.log(volume_proxy.clamp_min(1e-12))
    log_r = torch.log(mean_r.clamp_min(1e-12))
    slope = log_v / log_r                                 # [S]
    # Normalisation : courbure ~ (slope - D) / D
    D = points.shape[-1]
    curvature = (slope - D) / D

    return {
        "mean_curvature": float(curvature.mean().item()),
        "std_curvature": float(curvature.std().item()),
        "median_curvature": float(curvature.median().item()),
    }


# ---------------------------------------------------------------------------
# 3. Distorsion
# ---------------------------------------------------------------------------
def distortion(
    original: torch.Tensor,
    embedded: torch.Tensor,
    sample_size: int = 512,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Distorsion entre distances originales et plongées.

    distorsion = (max ratio) / (min ratio), 1.0 = isométrie parfaite.
    """
    N = original.shape[0]
    if N < 2:
        return 1.0

    i = torch.randint(0, N, (sample_size,), generator=generator)
    j = torch.randint(0, N, (sample_size,), generator=generator)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 1.0

    d_orig = torch.linalg.vector_norm(original[i] - original[j], dim=-1)
    d_emb = torch.linalg.vector_norm(embedded[i] - embedded[j], dim=-1)

    valid = d_orig > 1e-6
    if not valid.any():
        return 1.0
    ratios = d_emb[valid] / d_orig[valid]
    return float((ratios.max() / ratios.min()).item())


# ---------------------------------------------------------------------------
# 4. Rapport complet
# ---------------------------------------------------------------------------
def geometry_report(
    embeddings: torch.Tensor,
    manifold=None,
    n_quadruples: int = 512,
    n_samples_curvature: int = 256,
) -> Dict[str, float]:
    """Rapport complet des diagnostics géométriques.

    :param embeddings: [N, D] points sur la variété
    :param manifold: optionnel, pour proj
    :return: dict avec toutes les mesures
    """
    if manifold is not None:
        embeddings = manifold.proj(embeddings)

    emb = embeddings.detach().cpu()

    delta = delta_hyperbolicity(emb, n_quadruples=n_quadruples)
    curv_probe = curvature_probe(emb)
    curv_topo = curvature_from_distances(emb)
    curv_local = local_curvature_estimate(emb, n_samples=n_samples_curvature)

    report = {
        "delta_hyperbolicity": delta,
        "curvature_probe": curv_probe,
        "curvature_toponogov": curv_topo,
        "mean_curvature_local": curv_local["mean_curvature"],
        "std_curvature_local": curv_local["std_curvature"],
        "median_curvature_local": curv_local["median_curvature"],
        "mean_norm": float(torch.linalg.vector_norm(emb, dim=-1).mean()),
        "max_norm": float(torch.linalg.vector_norm(emb, dim=-1).max()),
        "std_norm": float(torch.linalg.vector_norm(emb, dim=-1).std()),
    }
    return report

# mccl/src/geometry/diagnostics.py — ajout

def tangent_alignment(
    points: torch.Tensor,
    manifold,
    sample_size: int = 256,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Fraction de points proches du bord de la variété.

    Utile pour vérifier la stabilité en courbure négative : si une grande
    fraction des points est proche du bord de la boule, l'entraînement est
    instable.

    :param points: [N, D] points sur la variété
    :param manifold: Hyperbolic ou ProductManifold
    :param sample_size: nombre de points à échantillonner
    :return: fraction dans [0, 1] (0 = aucun point au bord)
    """
    from .manifolds import Hyperbolic, ProductManifold

    if points.numel() == 0:
        return 0.0

    # Sous-échantillonnage
    N = points.shape[0]
    if N > sample_size:
        idx = torch.randint(0, N, (sample_size,), generator=generator)
        pts = points[idx]
    else:
        pts = points

    # Cas d'une variété unique hyperbolique
    if isinstance(manifold, Hyperbolic):
        norm = torch.linalg.vector_norm(pts, dim=-1)
        threshold = 0.9 * manifold.max_norm
        return float((norm > threshold).float().mean().item())

    # Cas d'un produit : moyenne pondérée des composantes hyperboliques
    if isinstance(manifold, ProductManifold):
        fractions = []
        for sl, m in zip(manifold.slices, manifold.manifolds):
            if isinstance(m, Hyperbolic):
                xi = pts[..., sl]
                norm = torch.linalg.vector_norm(xi, dim=-1)
                threshold = 0.9 * m.max_norm
                fractions.append(float((norm > threshold).float().mean().item()))
        if not fractions:
            return 0.0
        return sum(fractions) / len(fractions)

    return 0.0