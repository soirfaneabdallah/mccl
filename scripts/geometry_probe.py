# mccl/scripts/geometry_probe.py
"""Étape 0 : diagnostics géométriques sur features DINOv2 sans entraînement.

But : vérifier si la structure hyperbolique / sphérique / euclidienne
émerge déjà des features gelées avant tout apprentissage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mccl.models.backbone import DinoV2Backbone
from mccl.geometry.manifolds import Hyperbolic, Sphere, Euclidean, ProductManifold
from mccl.geometry.diagnostics import (
    delta_hyperbolicity,
    curvature_probe,
    distortion,
)
from mccl.utils.logging import get_logger
from mccl.utils.seed import set_seed

log = get_logger(__name__)


@torch.no_grad()
def extract_features(backbone, loader, device: str) -> torch.Tensor:
    """Renvoie la moyenne des tokens (pooling global) pour chaque image."""
    backbone.eval()
    feats = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        tokens = backbone(x)                  # [B, N, D]
        feats.append(tokens.mean(dim=1).cpu())  # [B, D]
    return torch.cat(feats, dim=0)


def random_loader(n: int, batch: int, image_size: int = 224):
    """Loader factice pour smoke-test (à remplacer par le vrai DataLoader)."""
    for _ in range(0, n, batch):
        b = min(batch, n - _)
        yield torch.randn(b, 3, image_size, image_size), torch.zeros(b, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sonde géométrique DINOv2.")
    ap.add_argument("--backbone", default="dinov2-small")
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/geometry_probe.json")
    args = ap.parse_args()

    set_seed(args.seed)
    log.info("Chargement %s (blocs=%d) sur %s", args.backbone, args.num_blocks, args.device)

    bb = DinoV2Backbone(name=args.backbone, num_blocks=args.num_blocks, frozen=True)
    bb.to(args.device)

    loader = random_loader(args.n_samples, args.batch)
    feats = extract_features(bb, loader, args.device)   # [N, D]
    log.info("Features extraites : %s", tuple(feats.shape))

    D = feats.shape[-1]
    scale = 0.1  # facteur d'échelle pour rester dans la boule

    results: dict[str, float] = {}

    # --- Hyperbolique ---
    m_h = Hyperbolic(dim=D, curvature=1.0)
    pts_h = m_h.expmap0(feats * scale)
    results["delta_hyperbolicity_H"] = delta_hyperbolicity(pts_h)
    results["distortion_H"] = distortion(feats, pts_h)
    results["mean_norm_H"] = curvature_probe(pts_h)

    # --- Sphère ---
    m_s = Sphere(dim=D)
    v0 = torch.zeros_like(feats)
    v0[..., 0] = 0.0
    pts_s = m_s.expmap0(feats * scale + v0)
    results["distortion_S"] = distortion(feats, pts_s)
    results["mean_norm_S"] = float(torch.linalg.vector_norm(pts_s, dim=-1).mean())

    # --- Euclidien ---
    m_e = Euclidean(dim=D)
    pts_e = m_e.expmap0(feats * scale)
    results["distortion_E"] = distortion(feats, pts_e)

    # --- Produit mixte (D/3 chacune) ---
    d3 = D // 3
    m_p = ProductManifold([
        Euclidean(d3),
        Hyperbolic(d3, curvature=1.0),
        Sphere(d3 + (D - 3 * d3)),  # ajuste pour retomber sur D
    ])
    pts_p = m_p.expmap0(feats * scale)
    results["delta_hyperbolicity_P"] = delta_hyperbolicity(pts_p)
    results["distortion_P"] = distortion(feats, pts_p)

    for k, v in results.items():
        log.info("%-28s = %.6f", k, v)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    log.info("Résultats écrits dans %s", out)


if __name__ == "__main__":
    main()