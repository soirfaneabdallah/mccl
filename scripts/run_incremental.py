# mccl/scripts/run_incremental.py
"""Lance une stratégie continue sur plusieurs ordres de classes et graines.

Boucle : pour chaque ordre × graine,
  - construire TaskStream (tâche 1 = 2 classes, puis +1 par étape),
  - entraîner la tâche courante selon la stratégie,
  - évaluer après chaque tâche (classes vues, OOD, diagnostics),
  - sauvegarder la matrice d'accuracy et les métriques.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mccl.config import load_config, ModelCfg, DataCfg, ContinualCfg, ExperimentCfg
from mccl.data.splits import load_splits
from mccl.data.transforms import build_train_transform, build_eval_transform
from mccl.data.incremental import TaskStream
from mccl.models.jepa import JepaModel
from mccl.continual.strategies import (
    Finetune, Joint, Replay, EWCMethod, DualMemory,
)
from mccl.training.losses import jepa_loss, classification_loss
from mccl.training.optim import build_optimizer
from mccl.evaluation.evaluator import evaluate
from mccl.evaluation.continual_metrics import (
    average_accuracy, forgetting, backward_transfer,
)
from mccl.utils.logging import get_logger
from mccl.utils.seed import set_seed

log = get_logger(__name__)


def build_strategy(name: str, cont_cfg: dict, model: JepaModel):
    if name == "finetune":
        return Finetune()
    if name == "joint":
        return Joint()
    if name == "replay":
        return Replay(capacity=cont_cfg.get("capacity", 512))
    if name == "ewc":
        return EWCMethod()
    if name == "dual_memory":
        return DualMemory(
            fast_cfg=cont_cfg.get("fast", {}),
            slow_cfg=cont_cfg.get("slow", {}),
            surprise_cfg=cont_cfg.get("surprise", {}),
            consolidation_cfg=cont_cfg.get("consolidation", {}),
        )
    raise ValueError(f"Stratégie inconnue : {name}")


def make_loader(paths, labels, indices, transform, batch_size, num_workers=0, shuffle=False):
    class _DS(torch.utils.data.Dataset):
        def __init__(self, paths, labels, indices, transform):
            self.paths, self.labels, self.indices, self.transform = paths, labels, indices, transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            from PIL import Image
            j = self.indices[i]
            img = Image.open(self.paths[j]).convert("RGB")
            return self.transform(img), int(self.labels[j])
    ds = _DS(paths, labels, indices, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def run_one(model_cfg_path, data_cfg_path, cont_cfg_path,
            order, seed, out_dir):
    set_seed(seed)
    data_cfg = DataCfg(**load_config(data_cfg_path))
    model_cfg = ModelCfg(**load_config(model_cfg_path))
    cont_cfg_raw = load_config(cont_cfg_path)

    splits = load_splits(f"{data_cfg.splits_dir}/waste5.json")
    paths = splits["paths"]
    labels = np.array(splits["labels"])
    tr_idx = np.array(splits["train"])
    va_idx = np.array(splits["val"])
    te_idx = np.array(splits["test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JepaModel(model_cfg).to(device)
    strategy = build_strategy(cont_cfg_raw["name"], cont_cfg_raw, model)

    stream = TaskStream(labels, order)
    acc_matrix = np.full((len(order), len(order)), np.nan)
    reports = []

    for task in stream:
        log.info("=== Tâche %d | classes=%s | seed=%d ===", task.index, task.classes, seed)
        strategy.begin_task(task)

        # Enregistre les nouvelles classes dans le classifieur
        for c in task.classes:
            model.classifier.add_class(c)

        # Entraînement (boucle simplifiée : 1 époque)
        train_loader = make_loader(
            paths, labels, np.intersect1d(task.train_idx, tr_idx),
            build_train_transform(data_cfg.image_size),
            data_cfg.batch_size, data_cfg.num_workers, shuffle=True,
        )
        opt = build_optimizer(model, lr=1e-3)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_cls = classification_loss(out.logits, y) if out.logits is not None else torch.tensor(0.0, device=device)
            loss = loss_cls
            opt.zero_grad(); loss.backward(); opt.step()
            strategy.observe((x, y))

        strategy.end_task(task)

        # Évaluation sur les classes vues
        eval_loader = make_loader(
            paths, labels, np.intersect1d(task.test_idx, te_idx),
            build_eval_transform(data_cfg.image_size),
            data_cfg.batch_size, data_cfg.num_workers, shuffle=False,
        )
        rep = evaluate(model, eval_loader, task.classes, task.index)
        reports.append({"task": task.index, "classes": task.classes,
                        "acc": rep.acc_overall})
        log.info("Tâche %d — acc=%.4f", task.index, rep.acc_overall)

        # Remplit la ligne task.index de la matrice d'accuracy
        for j, c in enumerate(task.classes):
            acc_matrix[task.index, j] = rep.acc_per_class.get(c, float("nan"))

    summary = {
        "order": order,
        "seed": seed,
        "avg_acc": average_accuracy(acc_matrix),
        "forgetting": forgetting(acc_matrix),
        "bwt": backward_transfer(acc_matrix),
        "acc_matrix": acc_matrix.tolist(),
        "reports": reports,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"run_o{order}_s{seed}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Résumé écrit : %s", out / f"run_o{order}_s{seed}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lance l'expérience incrémentale.")
    ap.add_argument("--exp", default="configs/experiment/incr_waste5_mixed_dual.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    exp = load_config(args.exp)
    e = exp["experiment"]
    for order in e["orders"]:
        for seed in e["seeds"]:
            run_one(
                model_cfg_path=f"configs/model/{exp['defaults'][1]['/model']}.yaml",
                data_cfg_path=f"configs/data/{exp['defaults'][0]['/data']}.yaml",
                cont_cfg_path=f"configs/continual/{exp['defaults'][2]['/continual']}.yaml",
                order=order, seed=seed, out_dir=args.out,
            )


if __name__ == "__main__":
    main()