# mccl/src/mccl/data/loader.py
"""Adaptation de Load_data pour MCCL.

Deux usages :
- lire un dataset.zip produit par Load_data.save() ;
- produire un index léger (chemin, label, groupe) sans charger les pixels.

Le chargement des pixels se fait à la demande via ImagePathDataset.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from mccl.utils.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".bmp", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Lecture directe d'un dataset.zip (format Load_data.save)
# ---------------------------------------------------------------------------
def load_pixels_from_zip(zip_path: str) -> Tuple[np.ndarray, List, dict]:
    """Lit un dataset.zip au format {'images.npy', 'labels.json', 'meta.json'}.

    Renvoie (images, labels, meta).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("images.npy") as f:
            images = np.load(io.BytesIO(f.read()), allow_pickle=False)
        with zf.open("labels.json") as f:
            labels = json.load(f)
        with zf.open("meta.json") as f:
            meta = json.load(f)
    return images, labels, meta


def dataframe_from_zip(zip_path: str) -> pd.DataFrame:
    """Reconstruit un DataFrame compatible avec celui de Load_data."""
    images, labels, meta = load_pixels_from_zip(zip_path)
    df = pd.DataFrame({
        "Image": list(images),
        "Label": pd.array(labels, dtype=object),
    })
    df.attrs["meta"] = meta
    return df


# ---------------------------------------------------------------------------
# Index léger (sans pixels)
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    path: str
    label: str
    group: str
    width: int
    height: int
    channels: int


class MCCLImageLoader:
    """Construit un index (chemin, label, groupe, dims) à partir d'un dossier.

    Contrairement à Load_data qui charge les pixels en RAM immédiatement,
    cette classe ne stocke que les chemins. Le chargement se fait plus tard
    dans ImagePathDataset.
    """

    def __init__(
        self,
        root: str,
        class_to_idx: dict,
        group_key: str = "source_id",
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = dict(class_to_idx)
        self.group_key = group_key
        self.recursive = recursive
        self.index: List[ImageRecord] = []

    def _iter_files(self, folder: Path) -> Iterable[Path]:
        if self.recursive:
            for dirpath, _, filenames in os.walk(folder):
                for f in sorted(filenames):
                    if f.lower().endswith(IMAGE_EXTS):
                        yield Path(dirpath) / f
        else:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(IMAGE_EXTS):
                    yield folder / f

    def _group_of(self, path: Path, cls_name: str) -> str:
        """Détermine le groupe anti-fuite.

        Convention par défaut : préfixe avant le premier '_' du nom de fichier.
        Adaptez selon votre dataset.
        """
        stem = path.stem
        if "_" in stem:
            return stem.split("_")[0]
        return f"{cls_name}"

    def scan(self) -> "MCCLImageLoader":
        from PIL import Image

        if not self.root.exists():
            raise FileNotFoundError(f"Racine introuvable : {self.root}")

        for cls_name in self.class_to_idx:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                log.warning("Classe absente : %s", cls_dir)
                continue
            for p in self._iter_files(cls_dir):
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        c = len(im.getbands())
                except Exception as e:
                    log.warning("Image illisible %s : %s", p, e)
                    continue

                self.index.append(ImageRecord(
                    path=str(p),
                    label=cls_name,
                    group=self._group_of(p, cls_name),
                    width=w,
                    height=h,
                    channels=c,
                ))

        log.info("Index construit : %d images, %d classes",
                 len(self.index), len(self.class_to_idx))
        return self

    @staticmethod
    def _hash_file(path: str, chunk: int = 1 << 20) -> str:
        import hashlib
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def deduplicate(self) -> "MCCLImageLoader":
        seen: dict = {}
        keep: List[ImageRecord] = []
        n_dup = 0
        for rec in self.index:
            h = self._hash_file(rec.path)
            if h in seen:
                n_dup += 1
                continue
            seen[h] = 1
            keep.append(rec)
        log.info("Doublons retirés : %d (restant %d)", n_dup, len(keep))
        self.index = keep
        return self

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "path": r.path,
                "Label": r.label,
                "group": r.group,
                "width": r.width,
                "height": r.height,
                "channels": r.channels,
            }
            for r in self.index
        ]
        return pd.DataFrame(rows)

    def save_index(self, json_path: str) -> None:
        payload = {
            "root": str(self.root),
            "class_to_idx": self.class_to_idx,
            "group_key": self.group_key,
            "records": [r.__dict__ for r in self.index],
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Index écrit : %s (%d entrées)", json_path, len(self.index))

    @staticmethod
    def load_index(json_path: str) -> "MCCLImageLoader":
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        obj = MCCLImageLoader(
            root=d["root"],
            class_to_idx=d["class_to_idx"],
            group_key=d.get("group_key", "source_id"),
        )
        obj.index = [ImageRecord(**r) for r in d["records"]]
        return obj

    def load_pixels(self, i: int, image_size: int, channels: int = 3) -> np.ndarray:
        from PIL import Image

        rec = self.index[i]
        with Image.open(rec.path) as im:
            im = im.convert("L" if channels == 1 else "RGB")
            im = im.resize((image_size, image_size))
            arr = np.array(im, dtype=np.uint8)
        if channels == 1:
            arr = np.expand_dims(arr, axis=-1)
        return arr

    def __len__(self) -> int:
        return len(self.index)


import os  # noqa: E402  (utilisé dans _iter_files)