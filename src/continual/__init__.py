# mccl/src/mccl/continual/__init__.py
"""Stratégies d'apprentissage continu pour MCCL."""

from .memory import FastMemory, SlowMemory
from .surprise import SurpriseScorer
from .consolidation import Consolidator
from .ewc import EWC
from .strategies import (
    Strategy,
    Finetune,
    Joint,
    Replay,
    EWCMethod,
    DualMemory,
    build_strategy,
)

__all__ = [
    "FastMemory", "SlowMemory",
    "SurpriseScorer",
    "Consolidator",
    "EWC",
    "Strategy", "Finetune", "Joint", "Replay", "EWCMethod", "DualMemory",
    "build_strategy",
]