# mccl/src/mccl/config.py
"""Dataclasses de configuration, chargement YAML et validation.

Structure :
- BackboneCfg, ManifoldCfg, PredictorCfg, ClassifierCfg, RegCfg : sous-configs
- ModelCfg : configuration complète du modèle
- DataCfg : données et chemins
- ContinualCfg : stratégie continue
- ExperimentCfg : expérience (ordres, graines, options)
- load_config, load_model_config, etc. : helpers
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
@dataclass
class BackboneCfg:
    """Configuration du backbone DINOv2."""
    name: str = "dinov2_vits14"  
    num_blocks: int = 10
    frozen: bool = True


@dataclass
class ManifoldCfg:
    """Configuration de la variété produit."""
    type: str = "product"
    components: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "euclidean", "dim": 64},
        {"type": "hyperbolic", "dim": 64, "curvature": 1.0},
        {"type": "sphere", "dim": 64},
    ])
    pooling: str = "tangent_mean"   # 'tangent_mean' | 'mean' | 'cls'


@dataclass
class PredictorCfg:
    """Configuration du prédicteur JEPA."""
    hidden: int = 384
    depth: int = 2


@dataclass
class ClassifierCfg:
    """Configuration du classifieur à prototypes."""
    type: str = "prototype"
    temperature: float = 1.0


@dataclass
class RegCfg:
    """Poids des régulariseurs géométriques."""
    lambda_h: float = 1.0e-3
    lambda_s: float = 1.0e-3
    lambda_e: float = 0.0


@dataclass
class JepaCfg:
    """Paramètres du masquage JEPA."""
    n_context: int = 64
    n_target: int = 32
    multi_step: int = 2


@dataclass
class ModelCfg:
    """Configuration complète du modèle."""
    name: str = "mixed"
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    manifold: ManifoldCfg = field(default_factory=ManifoldCfg)
    predictor: PredictorCfg = field(default_factory=PredictorCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    reg: RegCfg = field(default_factory=RegCfg)
    jepa: JepaCfg = field(default_factory=JepaCfg)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
@dataclass
class DataCfg:
    """Configuration des données."""
    name: str = "waste5"
    root: str = "data/raw"
    classes: List[str] = field(default_factory=lambda: [
        "Organique", "Textile", "Verre", "Plastique", "Canette",
    ])
    num_classes: int = 5
    splits_dir: str = "data/splits"
    ood_dir: str = "data/ood"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0
    group_key: str = "source_id"
    val_frac: float = 0.15
    test_frac: float = 0.15
    class_to_idx: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Apprentissage continu
# ---------------------------------------------------------------------------
@dataclass
class ContinualCfg:
    """Configuration d'une stratégie continue."""
    name: str = "dual_memory"

    # Fast memory
    fast_capacity: int = 256
    fast_strategy: str = "reservoir"

    # Slow memory
    slow_capacity_per_class: int = 32
    slow_strategy: str = "herding"

    # Surprise
    surprise_metric: str = "jepa_pred_error"
    surprise_normalize: str = "running_std"
    surprise_alpha: float = 0.99
    surprise_quantile: float = 0.9

    # Consolidation
    consolidation_trigger: str = "surprise"
    consolidation_epochs: int = 1
    consolidation_lr: float = 1.0e-3

    # EWC
    ewc_lambda: float = 1.0e4
    ewc_fisher_samples: int = 512

    # Replay
    replay_capacity: int = 512
    replay_ratio: float = 1.0

    # Finetune / Joint
    lr: float = 1.0e-3
    epochs_per_task: int = 5


# ---------------------------------------------------------------------------
# Expérience
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCfg:
    """Configuration d'une expérience complète."""
    name: str = "incr_waste5_mixed_dual"
    orders: List[List[int]] = field(default_factory=lambda: [
        [0, 1, 2, 3, 4],
        [0, 4, 1, 3, 2],
    ])
    seeds: List[int] = field(default_factory=lambda: [0])
    eval_after_each_task: bool = True
    eval_ood: bool = True
    eval_geometry: bool = True
    save_dir: str = "results"
    n_epochs: int = 3
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Chargement YAML
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    """Charge un fichier YAML en dictionnaire."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    # Supprime les éventuels en-têtes '# mccl/...'
    data.pop("defaults", None)
    return data


def load_config(path: str) -> Dict[str, Any]:
    """Charge un YAML brut en dictionnaire."""
    return _load_yaml(path)


def load_model_config(path: str) -> ModelCfg:
    """Charge un YAML de modèle et le convertit en ModelCfg.

    Gère les sous-dictionnaires (backbone, manifold, etc.).
    """
    d = _load_yaml(path)

    backbone = BackboneCfg(**d.get("backbone", {}))
    manifold = ManifoldCfg(**d.get("manifold", {}))
    predictor = PredictorCfg(**d.get("predictor", {}))
    classifier = ClassifierCfg(**d.get("classifier", {}))
    reg = RegCfg(**d.get("reg", {}))
    jepa = JepaCfg(**d.get("jepa", {}))

    cfg = ModelCfg(
        name=d.get("name", "mixed"),
        backbone=backbone,
        manifold=manifold,
        predictor=predictor,
        classifier=classifier,
        reg=reg,
        jepa=jepa,
    )
    validate_model_cfg(cfg)
    return cfg


def load_data_config(path: str) -> DataCfg:
    d = _load_yaml(path)
    # Suppression des clés non-utilisées si présentes
    allowed = {f.name for f in DataCfg.__dataclass_fields__.values()}
    d = {k: v for k, v in d.items() if k in allowed}
    return DataCfg(**d)


def load_continual_config(path: str) -> ContinualCfg:
    d = _load_yaml(path)
    allowed = {f.name for f in ContinualCfg.__dataclass_fields__.values()}
    d = {k: v for k, v in d.items() if k in allowed}
    return ContinualCfg(**d)


def load_experiment_config(path: str) -> ExperimentCfg:
    d = _load_yaml(path)
    if "experiment" in d:
        d = d["experiment"]
    allowed = {f.name for f in ExperimentCfg.__dataclass_fields__.values()}
    d = {k: v for k, v in d.items() if k in allowed}
    return ExperimentCfg(**d)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_model_cfg(cfg: ModelCfg) -> None:
    """Vérifie la cohérence d'une ModelCfg."""
    allowed_types = {"euclidean", "hyperbolic", "sphere"}

    if not cfg.manifold.components:
        raise ValueError("ManifoldCfg.components ne peut pas être vide.")

    for c in cfg.manifold.components:
        if "type" not in c or "dim" not in c:
            raise ValueError(
                f"Composante invalide : {c}. Il faut au moins 'type' et 'dim'."
            )
        if c["type"] not in allowed_types:
            raise ValueError(f"Type de variété inconnu : {c['type']}")
        if c["type"] == "hyperbolic":
            curv = c.get("curvature", 1.0)
            if curv <= 0:
                raise ValueError("curvature doit être > 0 pour hyperbolique.")
        if c["type"] == "sphere" and c["dim"] < 2:
            raise ValueError("Sphere.dim doit être >= 2.")
        if c["dim"] < 1:
            raise ValueError(f"dim doit être >= 1 pour {c['type']}.")

    if cfg.manifold.pooling not in {"tangent_mean", "mean", "cls"}:
        raise ValueError(
            f"pooling inconnu : {cfg.manifold.pooling}. "
            "Attendu : tangent_mean, mean, cls."
        )

    if cfg.backbone.num_blocks < 1:
        raise ValueError("backbone.num_blocks doit être >= 1.")

    if cfg.predictor.depth < 1:
        raise ValueError("predictor.depth doit être >= 1.")

    log.info(
        "ModelCfg validée : %s | backbone=%s (%d blocs, frozen=%s) | "
        "manifold=%d dims",
        cfg.name, cfg.backbone.name, cfg.backbone.num_blocks,
        cfg.backbone.frozen,
        sum(c["dim"] for c in cfg.manifold.components),
    )


def validate_data_cfg(cfg: DataCfg) -> None:
    """Vérifie la cohérence d'une DataCfg."""
    if cfg.num_classes != len(cfg.classes):
        raise ValueError(
            f"num_classes ({cfg.num_classes}) != len(classes) ({len(cfg.classes)})"
        )
    if cfg.val_frac + cfg.test_frac >= 1.0:
        raise ValueError("val_frac + test_frac doit être < 1.0")
    if cfg.image_size < 32:
        raise ValueError("image_size doit être >= 32")


# ---------------------------------------------------------------------------
# Conversion en dict
# ---------------------------------------------------------------------------
def cfg_to_dict(cfg) -> Dict[str, Any]:
    """Convertit une dataclass de config en dictionnaire."""
    return asdict(cfg)


def dict_to_cfg(d: Dict[str, Any], cfg_cls):
    """Reconstruit une dataclass depuis un dictionnaire."""
    return cfg_cls(**d)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------
def print_model_cfg(cfg: ModelCfg) -> None:
    """Affiche proprement une ModelCfg."""
    print("=" * 60)
    print(f"ModelCfg : {cfg.name}")
    print("=" * 60)
    print(f"Backbone   : {cfg.backbone.name} "
          f"({cfg.backbone.num_blocks} blocs, frozen={cfg.backbone.frozen})")
    print(f"Manifold   : {cfg.manifold.type} "
          f"(pooling={cfg.manifold.pooling})")
    for c in cfg.manifold.components:
        extra = f", curvature={c['curvature']}" if c["type"] == "hyperbolic" else ""
        print(f"  - {c['type']:10s} dim={c['dim']}{extra}")
    total_dim = sum(c["dim"] for c in cfg.manifold.components)
    print(f"  → dimension totale : {total_dim}")
    print(f"Predictor  : hidden={cfg.predictor.hidden}, depth={cfg.predictor.depth}")
    print(f"Classifier : {cfg.classifier.type} (T={cfg.classifier.temperature})")
    print(f"Reg        : λh={cfg.reg.lambda_h}, λs={cfg.reg.lambda_s}, λe={cfg.reg.lambda_e}")
    print(f"JEPA       : ctx={cfg.jepa.n_context}, tgt={cfg.jepa.n_target}, "
          f"steps={cfg.jepa.multi_step}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Presets rapides
# ---------------------------------------------------------------------------
def default_model_cfg() -> ModelCfg:
    """Configuration par défaut : E×H×S, 3×64 dims."""
    return ModelCfg()


def euclid_model_cfg(dim: int = 192) -> ModelCfg:
    """Configuration euclidienne pure (dim totale = 192)."""
    return ModelCfg(
        name="euclid",
        manifold=ManifoldCfg(
            components=[{"type": "euclidean", "dim": dim}],
            pooling="mean",
        ),
        reg=RegCfg(lambda_h=0.0, lambda_s=0.0, lambda_e=0.0),
    )


def hyper_model_cfg(dim: int = 192, curvature: float = 1.0) -> ModelCfg:
    """Configuration hyperbolique pure."""
    return ModelCfg(
        name="hyper_only",
        manifold=ManifoldCfg(
            components=[{"type": "hyperbolic", "dim": dim,
                         "curvature": curvature}],
            pooling="tangent_mean",
        ),
        reg=RegCfg(lambda_h=1.0e-3, lambda_s=0.0, lambda_e=0.0),
    )


def sphere_model_cfg(dim: int = 192) -> ModelCfg:
    """Configuration sphérique pure."""
    return ModelCfg(
        name="sphere_only",
        manifold=ManifoldCfg(
            components=[{"type": "sphere", "dim": dim}],
            pooling="tangent_mean",
        ),
        reg=RegCfg(lambda_h=0.0, lambda_s=1.0e-3, lambda_e=0.0),
    )