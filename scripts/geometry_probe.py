# mccl/scripts/geometry_probe.py
"""Sonde géométrique sur features DINOv2 gelées, sans entraînement.

Objectif : vérifier si la structure (hyperbolique / sphérique / euclidienne)
émerge déjà des features avant tout apprentissage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.geometry.manifolds import Euclidean, Sphere, Hyperbolic, ProductManifold
from src.geometry.diagnostics import (
    delta_hyperbolicity,
    curvature_probe,
    distortion,
    full_report,
)
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Extraction de features
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(backbone, loader, device: str, max_batches: int = None):
    """Renvoie les features globales (moyenne des tokens) pour chaque image."""
    backbone.eval()
    feats = []
    for i, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        tokens = backbone(x)                    # [B, N, D]
        feats.append(tokens.mean(dim=1).cpu())  # [B, D]
        if max_batches is not None and i + 1 >= max_batches:
            break
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Loader factice
# ---------------------------------------------------------------------------
def random_loader(n: int, batch: int, image_size: int = 224):
    for _ in range(0, n, batch):
        b = min(batch, n - _)
        yield torch.randn(b, 3, image_size, image_size), torch.zeros(b, dtype=torch.long)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    log.info("Backbone : %s | blocs : %d | device : %s",
             args.backbone, args.num_blocks, args.device)

    # --- Backbone ---
    from src.models.backbone import DinoV2Backbone
    bb = DinoV2Backbone(name=args.backbone, num_blocks=args.num_blocks,
                        frozen=True).to(args.device)

    # --- Features (synthétiques pour ce test) ---
    loader = random_loader(args.n_samples, args.batch)
    feats = extract_features(bb, loader, args.device)     # [N, D]
    log.info("Features : %s", tuple(feats.shape))

    D = feats.shape[-1]
    scale = 0.1

    # --- Diagnostics sur H, S, E, produit ---
    results = {}

    # Hyperbolique
    m_h = Hyperbolic(dim=D, curvature=1.0)
    pts_h = m_h.expmap0(feats * scale)
    results["delta_hyperbolicity_H"] = delta_hyperbolicity(pts_h, n_quadruples=512)
    results["curvature_probe_H"] = curvature_probe(pts_h)
    results["distortion_H"] = distortion(feats, pts_h, sample_size=512)

    # Sphère
    m_s = Sphere(dim=D)
    v0 = torch.zeros_like(feats)
    pts_s = m_s.expmap0(feats * scale + v0)
    results["distortion_S"] = distortion(feats, pts_s, sample_size=512)

    # Euclidien
    m_e = Euclidean(dim=D)
    pts_e = m_e.expmap0(feats * scale)
    results["distortion_E"] = distortion(feats, pts_e, sample_size=512)

    # Produit mixte (D/3 chacune)
    d3 = D // 3
    m_p = ProductManifold([
        Euclidean(d3),
        Hyperbolic(d3, curvature=1.0),
        Sphere(D - 2 * d3),
    ])
    pts_p = m_p.expmap0(feats * scale)
    results["delta_hyperbolicity_P"] = delta_hyperbolicity(pts_p, n_quadruples=512)
    results["distortion_P"] = distortion(feats, pts_p, sample_size=512)

    for k, v in results.items():
        log.info("%-28s = %.6f", k, v)

    # --- Écriture ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    log.info("Résultats écrits : %s", out)


if __name__ == "__main__":
    main()