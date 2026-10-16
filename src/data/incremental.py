# mccl/src/mccl/data/incremental.py  (version corrigée, à préférer)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Task:
    index: int
    classes: List[int]
    seen_classes: List[int]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_tasks(
    labels: np.ndarray,
    order: Sequence[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> List[Task]:
    """Construit la liste des tâches incrémentales.

    Tâche 0 : order[0] et order[1].
    Tâche t (t >= 1) : order[t+1].
    """
    labels = np.asarray(labels)
    order = list(order)
    tasks: List[Task] = []
    seen: List[int] = []

    for t in range(len(order) - 1):
        if t == 0:
            cls_set = [order[0], order[1]]
        else:
            cls_set = [order[t + 1]]
        seen = seen + cls_set

        train_mask = np.isin(labels[train_idx], cls_set)
        val_mask = np.isin(labels[val_idx], cls_set)
        test_mask = np.isin(labels[test_idx], seen)

        tasks.append(Task(
            index=t,
            classes=cls_set,
            seen_classes=list(seen),
            train_idx=train_idx[train_mask],
            val_idx=val_idx[val_mask],
            test_idx=test_idx[test_mask],
        ))
    return tasks


class TaskStream:
    """Itère sur les tâches incrémentales."""

    def __init__(self, labels, order, train_idx=None, val_idx=None, test_idx=None):
        labels = np.asarray(labels)
        n = len(labels)
        self.tasks = make_tasks(
            labels=labels,
            order=order,
            train_idx=train_idx if train_idx is not None else np.arange(n),
            val_idx=val_idx if val_idx is not None else np.arange(n),
            test_idx=test_idx if test_idx is not None else np.arange(n),
        )

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)