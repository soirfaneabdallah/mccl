# mccl/scripts/run_smoke_test.py
"""Smoke test : vérifie que le pipeline complet tourne sur un mini-batch.

- Charge un modèle JEPA avec un backbone gelé.
- Génère quelques images aléatoires.
- Entraîne 1 époque sur 2 classes.
- Évalue et affiche l'accuracy.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelCfg
from src.models.jepa import JepaModel
from src.training.losses import classification_loss
from src.training.optim import build_optimizer
from src.training.trainer import train_one_task, evaluate_accuracy
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device

    log.info("Device : %s", device)

    # --- Modèle ---
    cfg = ModelCfg()
    cfg.backbone.frozen = True
    model = JepaModel(cfg).to(device)
    model.register_new_classes([0, 1])
    log.info("Paramètres : %s", model.count_parameters())

    # --- Données aléatoires ---
    x = torch.randn(args.n, 3, 224, 224)
    y = torch.randint(0, 2, (args.n,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # --- Optimiseur ---
    opt = build_optimizer(model, lr=1e-3)

    # --- Loss ---
    def loss_fn(m, batch, out):
        _, yb = batch
        yb = yb.to(device)
        if out.logits is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return classification_loss(out.logits, yb)

    # --- Entraînement ---
    state = train_one_task(
        model=model, train_loader=loader,
        optimizer=opt, n_epochs=args.epochs,
        loss_fn=loss_fn, device=device,
    )

    # --- Évaluation ---
    acc = evaluate_accuracy(model, loader, device)
    log.info("Steps : %d", state.step)
    log.info("Accuracy finale : %.4f", acc)


if __name__ == "__main__":
    main()