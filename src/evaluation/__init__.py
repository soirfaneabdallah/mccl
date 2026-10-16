# mccl/src/mccl/evaluation/__init__.py
"""Métriques et évaluation pour MCCL."""

from .metrics import accuracy, per_class_accuracy, confusion_matrix
from .continual_metrics import (
    average_accuracy,
    forgetting,
    backward_transfer,
    forward_transfer,
)
from .ood_metrics import (
    auroc,
    fpr_at_95_tpr,
    energy_score,
    max_softmax_score,
    mahalanobis_score,
)
from .calibration import expected_calibration_error, reliability_diagram
from .evaluator import Evaluator, EvalReport

__all__ = [
    "accuracy", "per_class_accuracy", "confusion_matrix",
    "average_accuracy", "forgetting", "backward_transfer", "forward_transfer",
    "auroc", "fpr_at_95_tpr", "energy_score", "max_softmax_score",
    "mahalanobis_score",
    "expected_calibration_error", "reliability_diagram",
    "Evaluator", "EvalReport",
]