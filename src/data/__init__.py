# mccl/src/mccl/data/__init__.py
"""Chargement, découpage et augmentation des données pour MCCL."""

from .splits import stratified_group_split, save_splits, load_splits
from .incremental import Task, TaskStream
from .transforms import build_train_transform, build_eval_transform
from .masking import random_block_mask, apply_mask
from .ood import CORRUPTIONS, semantic_ood_classes, covariate_ood_transform
from .dataset import ImagePathDataset
from .loader import MCCLImageLoader, dataframe_from_zip

__all__ = [
    "stratified_group_split", "save_splits", "load_splits",
    "Task", "TaskStream",
    "build_train_transform", "build_eval_transform",
    "random_block_mask", "apply_mask",
    "CORRUPTIONS", "semantic_ood_classes", "covariate_ood_transform",
    "ImagePathDataset",
    "MCCLImageLoader", "dataframe_from_zip",
]