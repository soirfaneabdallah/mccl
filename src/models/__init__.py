# mccl/src/mccl/models/__init__.py
"""Modèles MCCL : backbone, tête variété, prédicteur, classifieur, JEPA."""

from .backbone import DinoV2Backbone
from .manifold_head import ManifoldHead
from .predictor import Predictor
from .classifier import PrototypeClassifier
from .jepa import JepaModel, JepaOutput

__all__ = [
    "DinoV2Backbone",
    "ManifoldHead",
    "Predictor",
    "PrototypeClassifier",
    "JepaModel",
    "JepaOutput",
]