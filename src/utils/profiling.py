# mccl/src/mccl/utils/profiling.py
"""Mesure du temps et de la mémoire (CPU/GPU)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

import torch


class Timer:
    """Chronomètre simple avec accumulation."""

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        dt = time.perf_counter() - self._t0
        self.total += dt
        self.n += 1
        self._t0 = None
        return dt

    @property
    def mean(self) -> float:
        return self.total / max(self.n, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.n = 0


@contextmanager
def timed(timer: Optional[Timer] = None):
    """Context manager : mesure une durée."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if timer is not None:
            timer.total += dt
            timer.n += 1


def gpu_memory_report() -> dict:
    """Renvoie l'état mémoire du GPU."""
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 ** 2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()