# mccl/src/mccl/utils/__init__.py
"""Utilitaires transverses : logging, seed, checkpoint, visualisation."""

from .logging import get_logger
from .seed import set_seed, worker_init_fn
from .checkpoint import save_ckpt, load_ckpt, save_json, load_json
from .viz import plot_accuracy_matrix, plot_embeddings_2d

__all__ = [
    "get_logger",
    "set_seed", "worker_init_fn",
    "save_ckpt", "load_ckpt", "save_json", "load_json",
    "plot_accuracy_matrix", "plot_embeddings_2d",
]