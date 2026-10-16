# mccl/src/mccl/data/dataset.py
"""Dataset PyTorch minimal basé sur une liste de chemins."""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    """Charge une image à la demande depuis son chemin.

    :param paths: liste de chemins
    :param labels: liste d'entiers (labels encodés)
    :param transform: transformation torchvision appliquée à l'image PIL
    :param channels: 1 (grayscale) ou 3 (RGB)
    """

    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: Callable | None = None,
        channels: int = 3,
    ) -> None:
        assert len(paths) == len(labels), "paths et labels doivent avoir même taille"
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.channels = channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = im.convert("L" if self.channels == 1 else "RGB")
        if self.transform is not None:
            x = self.transform(im)
        else:
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return x, int(self.labels[i])