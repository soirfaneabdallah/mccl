# mccl/src/mccl/data/splits.py
"""Splits figés train/val/test : stratifiés par classe ET par groupe.

Le découpage par groupe évite les fuites dues aux quasi-doublons
(même objet photographié plusieurs fois, même source).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def stratified_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Découpe train/val/test en stratifiant par classe et par groupe.

    Aucun groupe n'apparaît dans deux splits différents (anti-fuite).

    :param labels: array d'entiers, classe de chaque échantillon
    :param groups: array de strings ou d'entiers, identifiant de groupe
    :param val_frac: fraction de validation
    :param test_frac: fraction de test
    :param seed: graine aléatoire
    :return: (train_idx, val_idx, test_idx)
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        g_c = np.unique(groups[idx_c])
        rng.shuffle(g_c)

        n = len(g_c)
        if n < 3:
            # Trop peu de groupes : tout en train
            train.extend(idx_c.tolist())
            continue

        n_val = max(1, int(round(val_frac * n)))
        n_test = max(1, int(round(test_frac * n)))
        # Ajustement si on dépasse
        if n_val + n_test >= n:
            n_val = max(1, n // 5)
            n_test = max(1, n // 5)

        g_val = set(g_c[:n_val].tolist())
        g_test = set(g_c[n_val:n_val + n_test].tolist())

        for i in idx_c:
            gi = groups[i]
            if gi in g_val:
                val.append(i)
            elif gi in g_test:
                test.append(i)
            else:
                train.append(i)

    return (
        np.array(train, dtype=int),
        np.array(val, dtype=int),
        np.array(test, dtype=int),
    )


def save_splits(path: str, splits: Dict[str, np.ndarray]) -> None:
    """Écrit un dictionnaire de splits en JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v).tolist() for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: str) -> Dict[str, np.ndarray]:
    """Lit un fichier de splits JSON."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: np.array(v, dtype=int) for k, v in d.items()}


def check_no_leak(
    groups: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> None:
    """Vérifie qu'aucun groupe n'apparaît dans deux splits."""
    g_tr = set(groups[train].tolist())
    g_va = set(groups[val].tolist())
    g_te = set(groups[test].tolist())
    assert g_tr.isdisjoint(g_va), "Fuite train/val"
    assert g_tr.isdisjoint(g_te), "Fuite train/test"
    assert g_va.isdisjoint(g_te), "Fuite val/test"