# mccl/src/mccl/data/ood.py
"""OOD covariant (corruptions) et OOD sémantique (classes inconnues)."""
from __future__ import annotations

from typing import List

import torch
from torchvision import transforms as T


CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
]


class AddGaussianNoise:
    """Ajoute un bruit gaussien à un tenseur image."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class AddShotNoise:
    """Bruit de type shot noise (Poisson)."""

    def __init__(self, scale: float = 60.0):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.poisson(torch.clamp(x, 0.0, 1.0) * self.scale) / self.scale


def covariate_ood_transform(name: str, image_size: int = 224):
    """Renvoie une transformation OOD covariant pour un nom donné."""
    base = [
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ]

    if name == "gaussian_noise":
        return T.Compose(base + [AddGaussianNoise(0.1)])
    if name == "shot_noise":
        return T.Compose(base + [AddShotNoise(60.0)])
    if name == "brightness":
        return T.Compose([T.Resize(int(image_size * 1.14)),
                          T.CenterCrop(image_size),
                          T.ColorJitter(brightness=0.5),
                          T.ToTensor()])
    if name == "contrast":
        return T.Compose([T.Resize(int(image_size * 1.14)),
                          T.CenterCrop(image_size),
                          T.ColorJitter(contrast=0.5),
                          T.ToTensor()])
    if name == "defocus_blur":
        return T.Compose(base + [T.GaussianBlur(kernel_size=7, sigma=2.0)])
    if name == "motion_blur":
        return T.Compose(base + [T.GaussianBlur(kernel_size=(7, 1), sigma=(2.0, 0.1))])
    if name == "jpeg_compression":
        # JPEG compression simulée via PIL
        return T.Compose(base)  # à raffiner avec une transformation PIL custom

    # Défaut : identité (crop + normalisation ImageNet)
    return T.Compose(base)


def semantic_ood_classes(all_classes: List[str], seen: List[str]) -> List[str]:
    """Renvoie les classes non encore vues (OOD sémantique)."""
    return [c for c in all_classes if c not in seen]