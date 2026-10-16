# mccl/scripts/prepare_data.py (extrait modifié)

from pathlib import Path

from data.loader import MCCLImageLoader

def collect_metadata(root: Path, class_to_idx: dict[str, int],
                     group_key: str = "source_id") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Utilise MCCLImageLoader pour construire l'index, dédoublonner,
    et renvoyer (labels, groups, paths) prêts pour le split."""
    loader = (
        MCCLImageLoader(root=str(root), class_to_idx=class_to_idx,
                        group_key=group_key, recursive=True)
        .scan()
        .deduplicate()
    )
    labels = np.array([class_to_idx[r.label] for r in loader.index], dtype=int)
    groups = np.array([r.group for r in loader.index])
    paths = [r.path for r in loader.index]
    return labels, groups, paths