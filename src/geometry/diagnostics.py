# mccl/src/mccl/geometry/diagnostics.py
"""Diagnostics géométriques : δ-hyperbolicité, courbure, distorsion.

Ces mesures servent à vérifier que les features ont bien une structure
hiérarchique (hyperbolique) plutôt qu'euclidienne ou sphérique.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .manifolds import Hyperbolic


# ---------------------------------------------------------------------------
# δ-hyperbolicité (Gromov)
# ---------------------------------------------------------------------------
def delta_hyperbolicity(
    points: Tensor,
    n_quadruples: int = 1024,
    generator: torch.Generator | None = None,
) -> float:
    """Estimation Monte-Carlo de la δ-hyperbolicité de Gromov.

    Pour un espace δ-hyperbolique, pour tout quadruple (x,y,z,w) :
        d(x,y) + d(z,w) <= max(d(x,z)+d(y,w), d(x,w)+d(y,z)) + 2δ

    On calcule la moyenne des excès sur des quadruples aléatoires.

    :param points: tenseur [N, D] de points (dans l'espace ambient)
    :param n_quadruples: nombre de quadruples tirés
    :param generator: générateur torch pour la reproductibilité
    :return: estimation de δ (petit = hyperbolique, grand = euclidien)
    """
    N = points.shape[0]
    if N < 4:
        return 0.0

    idx = torch.randint(0, N, (n_quadruples, 4), generator=generator)
    x = points[idx[:, 0]]
    y = points[idx[:, 1]]
    z = points[idx[:, 2]]
    w = points[idx[:, 3]]

    def d(a: Tensor, b: Tensor) -> Tensor:
        return torch.linalg.vector_norm(a - b, dim=-1)

    s1 = d(x, y) + d(z, w)
    s2 = d(x, z) + d(y, w)
    s3 = d(x, w) + d(y, z)

    # Le plus grand des trois
    smax = torch.maximum(torch.maximum(s1, s2), s3)
    # Excès : smax - s1 (par rapport au pair opposé)
    # Formulation standard : δ = (s1 - smax) / 2 pour le plus grand écart
    delta = (smax - s1) / 2.0
    return float(delta.mean().item())


# ---------------------------------------------------------------------------
# Courbure estimée
# ---------------------------------------------------------------------------
def curvature_probe(points: Tensor) -> float:
    """Proxy grossier de courbure : norme moyenne des points.

    Dans la boule de Poincaré, des points proches du bord (norme proche
    de 1) indiquent une structure plus « hiérarchique ».
    """
    n = torch.linalg.vector_norm(points, dim=-1)
    return float(n.mean().item())


def curvature_from_distances(
    points: Tensor,
    n_triples: int = 512,
    generator: torch.Generator | None = None,
) -> float:
    """Estimation de courbure via la formule de Toponogov (approximative).

    Pour trois points formant un triangle, on compare la somme des
    angles à π. En courbure négative, la somme est < π.

    :return: valeur dans [-1, 1] (négatif = hyperbolique)
    """
    N = points.shape[0]
    if N < 3:
        return 0.0

    idx = torch.randint(0, N, (n_triples, 3), generator=generator)
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]

    def cos_angle(u, v, w):
        # angle en v
        vu = u - v
        vw = w - v
        dot = (vu * vw).sum(dim=-1)
        nu = torch.linalg.vector_norm(vu, dim=-1).clamp_min(1e-12)
        nw = torch.linalg.vector_norm(vw, dim=-1).clamp_min(1e-12)
        return (dot / (nu * nw)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    angle_a = torch.acos(cos_angle(b, a, c))
    angle_b = torch.acos(cos_angle(a, b, c))
    angle_c = torch.acos(cos_angle(a, c, b))

    angle_sum = angle_a + angle_b + angle_c
    # En euclidien : π ; en hyperbolique : < π ; en sphérique : > π
    deviation = (angle_sum - torch.pi).mean()
    return float(deviation.item())


# ---------------------------------------------------------------------------
# Distorsion
# ---------------------------------------------------------------------------
def distortion(
    original: Tensor,
    embedded: Tensor,
    sample_size: int = 512,
    generator: torch.Generator | None = None,
) -> float:
    """Distorsion entre distances originales et distances plongées.

    distorsion = (max ratio) / (min ratio) où ratio = d_embed / d_orig.

    :param original: [N, D1] points dans l'espace source
    :param embedded: [N, D2] points dans l'espace cible
    :param sample_size: nombre de paires à échantillonner
    :return: distorsion (1.0 = parfaitement isométrique)
    """
    N = original.shape[0]
    if N < 2:
        return 1.0

    # Échantillon de paires
    i = torch.randint(0, N, (sample_size,), generator=generator)
    j = torch.randint(0, N, (sample_size,), generator=generator)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 1.0

    d_orig = torch.linalg.vector_norm(original[i] - original[j], dim=-1)
    d_emb = torch.linalg.vector_norm(embedded[i] - embedded[j], dim=-1)

    # Éviter les divisions par zéro
    valid = d_orig > 1e-6
    if not valid.any():
        return 1.0

    ratios = d_emb[valid] / d_orig[valid]
    return float((ratios.max() / ratios.min()).item())


# ---------------------------------------------------------------------------
# Alignement tangent
# ---------------------------------------------------------------------------
def tangent_alignment(
    points: Tensor,
    manifold: Hyperbolic,
    sample_size: int = 256,
    generator: torch.Generator | None = None,
) -> float:
    """Mesure à quel point les points restent dans une zone stable.

    Renvoie la fraction de points dont la norme dépasse 0.9 * max_norm,
    c'est-à-dire proches du bord de la boule.

    :return: fraction dans [0, 1] (0 = aucun point au bord)
    """
    norm = torch.linalg.vector_norm(points, dim=-1)
    threshold = 0.9 * manifold.max_norm
    return float((norm > threshold).float().mean().item())


# ---------------------------------------------------------------------------
# Rapport complet
# ---------------------------------------------------------------------------
def full_report(
    points: Tensor,
    manifold=None,
    n_quadruples: int = 512,
) -> dict:
    """Rassemble tous les diagnostics en un dictionnaire.

    :param points: tenseur [N, D] de points
    :param manifold: optionnel, pour tangent_alignment
    """
    report = {
        "delta_hyperbolicity": delta_hyperbolicity(points, n_quadruples),
        "curvature_probe": curvature_probe(points),
        "curvature_from_distances": curvature_from_distances(points),
    }
    if isinstance(manifold, Hyperbolic):
        report["tangent_alignment"] = tangent_alignment(points, manifold)
    return report