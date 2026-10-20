# mccl/scripts/prepare_data.py
"""Prépare les données : dédoublonnage, splits figés (train/val/test).

Deux modes d'entrée :
- un dataset.zip produit par Load_data.save() (images.npy + labels.json + meta.json) ;
- un dossier d'images organisé en <classe>/*.jpg.

Sortie : un fichier data/splits/waste5.json contenant les indices des
splits train/val/test, ainsi que les métadonnées (classes, mapping).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dédoublonnage
# ---------------------------------------------------------------------------
def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Empreinte SHA-1 d'un fichier."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def deduplicate_paths(paths: List[str]) -> List[int]:
    """Retourne les indices à conserver après dédoublonnage par hash."""
    seen: Dict[str, int] = {}
    keep: List[int] = []
    n_dup = 0
    for i, p in enumerate(paths):
        h = file_hash(Path(p))
        if h in seen:
            n_dup += 1
            continue
        seen[h] = i
        keep.append(i)
    log.info("Dédoublonnage : %d doublons retirés sur %d", n_dup, len(paths))
    return keep


# ---------------------------------------------------------------------------
# Chargement depuis le zip
# ---------------------------------------------------------------------------
def load_from_zip(zip_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """Lit un dataset.zip. Renvoie (labels, groups, paths, meta).

    Le zip doit contenir : images.npy, labels.json, meta.json.
    """
    import io
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("images.npy") as f:
            images = np.load(io.BytesIO(f.read()), allow_pickle=False)
        with zf.open("labels.json") as f:
            labels = json.load(f)
        with zf.open("meta.json") as f:
            meta = json.load(f)

    paths = [f"index_{i}" for i in range(len(labels))]
    groups = np.array([f"g{i}" for i in range(len(labels))])
    return np.array(labels), groups, paths, meta


# ---------------------------------------------------------------------------
# Chargement depuis un dossier
# ---------------------------------------------------------------------------
def load_from_folder(root: str, class_to_idx: Dict[str, int],
                     group_key: str = "source_id"):
    """Parcourt un dossier <racine>/<classe>/*.jpg et renvoie les métadonnées."""
    from src.data.loader import MCCLImageLoader

    loader = (
        MCCLImageLoader(root=root, class_to_idx=class_to_idx,
                        group_key=group_key, recursive=True)
        .scan()
        .deduplicate()
    )
    labels = np.array([class_to_idx[r.label] for r in loader.index], dtype=int)
    groups = np.array([r.group for r in loader.index])
    paths = [r.path for r in loader.index]
    return labels, groups, paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Prépare splits figés.")
    ap.add_argument("--zip", default=None, help="chemin dataset.zip (Load_data)")
    ap.add_argument("--root", default=None, help="dossier racine des images")
    ap.add_argument("--classes", nargs="+",
                    default=["Organique", "Textile", "Verre", "Plastique", "Canette"])
    ap.add_argument("--out", default="data/splits/waste5.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    if args.zip is None and args.root is None:
        log.error("Fournir --zip ou --root.")
        return

    class_to_idx = {c: i for i, c in enumerate(args.classes)}

    # --- Chargement ---
    if args.zip:
        log.info("Chargement depuis zip : %s", args.zip)
        labels, groups, paths, meta = load_from_zip(args.zip)
        classes = meta.get("name_label_", args.classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        # Si labels sont des strings, encoder
        if isinstance(labels[0], str):
            labels = np.array([class_to_idx[l] for l in labels])
    else:
        log.info("Chargement depuis dossier : %s", args.root)
        labels, groups, paths = load_from_folder(
            args.root, class_to_idx, group_key="source_id")

    log.info("Images chargées : %d, classes : %d", len(labels), len(class_to_idx))

    # --- Dédoublonnage (dossier seulement, le zip est déjà traité) ---
    if not args.no_dedup and args.root:
        keep = deduplicate_paths(paths)
        labels = labels[keep]
        groups = groups[keep]
        paths = [paths[i] for i in keep]

    # --- Split stratifié par groupe ---
    from src.data.splits import stratified_group_split, check_no_leak

    tr, va, te = stratified_group_split(
        labels=labels, groups=groups,
        val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed,
    )
    check_no_leak(groups, tr, va, te)

    # --- Écriture ---
    payload = {
        "root": args.root or args.zip,
        "classes": list(class_to_idx.keys()),
        "class_to_idx": class_to_idx,
        "paths": list(paths),
        "labels": labels.tolist(),
        "groups": [str(g) for g in groups.tolist()],
        "train": tr.tolist(),
        "val": va.tolist(),
        "test": te.tolist(),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Splits écrits : %s (train=%d, val=%d, test=%d)",
             out, len(tr), len(va), len(te))


if __name__ == "__main__":
    main()