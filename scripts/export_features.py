# mccl/scripts/export_features.py
"""Extrait et sauvegarde les features d'un modèle entraîné.

Utile pour : analyse, visualisation (t-SNE), détection OOD
(Mahalanobis), rejeu en espace de features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.checkpoint import load_ckpt

log = get_logger(__name__)


@torch.no_grad()
def extract_features(model, loader, device):
    """Renvoie (features, labels) pour toutes les images du loader."""
    model.eval()
    feats_all = []
    labels_all = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        feats = out.global_embed if hasattr(out, "global_embed") else out
        feats_all.append(feats.cpu())
        labels_all.append(y)
    return torch.cat(feats_all, 0), torch.cat(labels_all, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="chemin du checkpoint")
    ap.add_argument("--out", default="results/features.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # Charger le modèle
    from mccl.config import ModelCfg
    from mccl.models.jepa import JepaModel

    cfg = ModelCfg()
    cfg.backbone.frozen = True
    model = JepaModel(cfg).to(args.device)
    meta = load_ckpt(args.ckpt, model, map_location=args.device)
    log.info("Checkpoint chargé : %s", meta)

    # Placeholder : données aléatoires
    x = torch.randn(64, 3, 224, 224)
    y = torch.randint(0, 5, (64,))
    loader = DataLoader(list(zip(x, y)), batch_size=8)

    feats, labels = extract_features(model, loader, args.device)
    log.info("Features : %s | Labels : %s", tuple(feats.shape), tuple(labels.shape))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=feats.numpy(), labels=labels.numpy())
    log.info("Features sauvegardées : %s", out)


if __name__ == "__main__":
    main()