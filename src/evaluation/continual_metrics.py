# mccl/src/mccl/evaluation/continual_metrics.py
"""Métriques d'apprentissage continu : accuracy moyenne, oubli, transferts."""
from __future__ import annotations

import numpy as np


def average_accuracy(acc_matrix: np.ndarray) -> float:
    """Accuracy moyenne après la dernière tâche.

    :param acc_matrix: matrice [T, T] où acc_matrix[i, j] = accuracy
        sur la classe j après avoir entraîné jusqu'à la tâche i.
        Les cellules non renseignées doivent être NaN.
    """
    T = acc_matrix.shape[0]
    last_row = acc_matrix[T - 1]
    return float(np.nanmean(last_row))


def forgetting(acc_matrix: np.ndarray) -> float:
    """Oubli moyen.

    Pour chaque classe j apprise à la tâche j (ou avant), on compare
    le maximum d'accuracy observé juste après l'avoir apprise
    (acc_matrix[j, j]) à l'accuracy finale (acc_matrix[T-1, j]).
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    drops = []
    for j in range(T - 1):
        best = acc_matrix[j, j]
        final = acc_matrix[T - 1, j]
        if not np.isnan(best) and not np.isnan(final):
            drops.append(best - final)
    return float(np.mean(drops)) if drops else 0.0


def backward_transfer(acc_matrix: np.ndarray) -> float:
    """Transfert arrière (BWT).

    Différence entre l'accuracy finale sur les classes passées et
    l'accuracy juste après les avoir apprises.
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    bwt = []
    for j in range(T - 1):
        learned = acc_matrix[j, j]
        final = acc_matrix[T - 1, j]
        if not np.isnan(learned) and not np.isnan(final):
            bwt.append(final - learned)
    return float(np.mean(bwt)) if bwt else 0.0


def forward_transfer(acc_matrix: np.ndarray, random_baseline: np.ndarray) -> float:
    """Transfert avant (FWT).

    Compare l'accuracy sur une nouvelle tâche juste après son
    apprentissage à un baseline aléatoire.

    :param random_baseline: array [T] avec l'accuracy d'un modèle
        aléatoire sur chaque tâche
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    fwt = []
    for j in range(1, T):
        after = acc_matrix[j, j]
        base = random_baseline[j]
        if not np.isnan(after) and not np.isnan(base):
            fwt.append(after - base)
    return float(np.mean(fwt)) if fwt else 0.0


def learning_accuracy(acc_matrix: np.ndarray) -> float:
    """Accuracy moyenne sur la diagonale (juste après apprentissage)."""
    T = acc_matrix.shape[0]
    diag = [acc_matrix[i, i] for i in range(T) if not np.isnan(acc_matrix[i, i])]
    return float(np.mean(diag)) if diag else 0.0