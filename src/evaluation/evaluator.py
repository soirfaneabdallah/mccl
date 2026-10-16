# mccl/src/mccl/evaluation/evaluator.py
"""Évaluateur complet : après chaque tâche, sur classes vues + OOD."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import accuracy, per_class_accuracy, confusion_matrix
from .ood_metrics import auroc, fpr_at_95_tpr, max_softmax_score, energy_score
from .calibration import expected_calibration_error


@dataclass
class EvalReport:
    """Rapport d'évaluation après une tâche."""
    task: int
    seen_classes: List[int]
    acc_overall: float = 0.0
    acc_per_class: Dict[int, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None
    ood_auroc_msp: float = 0.0
    ood_auroc_energy: float = 0.0
    ood_fpr95_msp: float = 0.0
    ood_fpr95_energy: float = 0.0
    ece: float = 0.0
    geo_diag: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


class Evaluator:
    """Évalue un modèle après chaque tâche.

    - accuracy globale et par classe sur les classes vues ;
    - matrice de confusion ;
    - détection OOD via max-softmax et energy ;
    - calibration (ECE) ;
    - diagnostics géométriques (optionnels).
    """

    def __init__(
        self,
        n_classes_max: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_classes_max = n_classes_max
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        seen_classes: List[int],
        task_idx: int,
        ood_loader: Optional[DataLoader] = None,
        geo_diag_fn: Optional[Callable] = None,
    ) -> EvalReport:
        """Évalue le modèle sur le loader courant et (optionnel) sur l'OOD."""
        model.eval()

        all_logits: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        all_feats: List[torch.Tensor] = []

        for batch in loader:
            x, y = batch
            x = x.to(self.device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
            if hasattr(out, "global_embed"):
                all_feats.append(out.global_embed.cpu())

        report = EvalReport(task=task_idx, seen_classes=list(seen_classes))

        if not all_logits:
            return report

        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        report.n_samples = int(y.size(0))

        # Accuracy globale et par classe
        report.acc_overall = accuracy(logits, y)
        report.acc_per_class = {
            c: float(v)
            for c, v in zip(
                range(self.n_classes_max),
                per_class_accuracy(logits, y, self.n_classes_max),
            )
            if not np.isnan(v)
        }
        report.confusion = confusion_matrix(logits, y, self.n_classes_max)

        # Calibration
        report.ece = expected_calibration_error(logits, y)

        # OOD
        if ood_loader is not None:
            id_scores_msp = max_softmax_score(logits)
            id_scores_energy = energy_score(logits)

            ood_logits: List[torch.Tensor] = []
            for batch in ood_loader:
                x_ood = batch[0].to(self.device)
                out = model(x_ood)
                ood_logits.append(
                    (out.logits if hasattr(out, "logits") else out).cpu()
                )
            if ood_logits:
                ood_logits = torch.cat(ood_logits, dim=0)
                ood_scores_msp = max_softmax_score(ood_logits)
                ood_scores_energy = energy_score(ood_logits)

                report.ood_auroc_msp = auroc(id_scores_msp, ood_scores_msp)
                report.ood_auroc_energy = auroc(id_scores_energy, ood_scores_energy)
                report.ood_fpr95_msp = fpr_at_95_tpr(id_scores_msp, ood_scores_msp)
                report.ood_fpr95_energy = fpr_at_95_tpr(
                    id_scores_energy, ood_scores_energy
                )

        # Diagnostics géométriques
        if geo_diag_fn is not None and all_feats:
            feats = torch.cat(all_feats, dim=0)
            try:
                report.geo_diag = geo_diag_fn(feats)
            except Exception as e:
                report.geo_diag = {"error": str(e)}

        return report

    def to_dict(self, report: EvalReport) -> dict:
        """Convertit un rapport en dictionnaire sérialisable."""
        return {
            "task": report.task,
            "seen_classes": list(report.seen_classes),
            "acc_overall": report.acc_overall,
            "acc_per_class": report.acc_per_class,
            "confusion": (
                report.confusion.tolist()
                if report.confusion is not None
                else None
            ),
            "ood_auroc_msp": report.ood_auroc_msp,
            "ood_auroc_energy": report.ood_auroc_energy,
            "ood_fpr95_msp": report.ood_fpr95_msp,
            "ood_fpr95_energy": report.ood_fpr95_energy,
            "ece": report.ece,
            "geo_diag": report.geo_diag,
            "n_samples": report.n_samples,
        }