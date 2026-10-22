# MCCL — Mixed-Curvature Continual Learning

Projet pilote de **classification incrémentale d'images de déchets** avec un encodeur de type JEPA dont l'espace latent est structuré comme une **variété produit à courbure mixte** (hyperbolique × sphérique × euclidienne) et un mécanisme d'**apprentissage continu à mémoire duale**.

Ce dépôt sert de banc d'essai à la thèse *« Modèles du monde à courbure mixte pour la planification hiérarchique en robotique »*. Il teste l'encodeur (hypothèse H1) et l'apprentissage continu (WP3). Il ne teste **pas** la planification hiérarchique (H2), faute d'actions et de sous-buts en classification d'images.

> **Statut** : phase de développement. 
---

## 1. Objectifs scientifiques

| # | Question | Mesure |
|---|----------|--------|
| Q1 | La courbure mixte réduit-elle la distorsion et améliore-t-elle la classification, à dimension latente égale, par rapport à un espace euclidien ? | Accuracy, distorsion, δ-hyperbolicité |
| Q2 | Chaque composante (H, S, R) capte-t-elle le type de structure attendu (hiérarchie, périodicité, grandeurs continues) ? | Sondes linéaires par composante |
| Q3 | La mémoire duale limite-t-elle l'oubli par rapport à EWC, au rejeu seul et au fine-tuning naïf ? | Oubli moyen, transfert arrière |
| Q4 | Le signal de surprise détecte-t-il les entrées hors distribution ? | AUROC, FPR95 |
| Q5 | Le modèle reste-t-il calibré et robuste hors distribution au fil des tâches ? | ECE, accuracy sous corruption |

Les résultats négatifs (données peu hyperboliques, pas de gain de la courbure mixte) sont des résultats valides et seront rapportés.

## 2. Protocole expérimental

**Données** : 5 classes de déchets, découpage train / validation / test **par groupe ou source** (anti-fuite entre quasi-doublons), indices figés dans `data/splits/`.

**Scénario incrémental par classes** :

```
Tâche 1 : 2 classes          -> entraînement, évaluation
Tâche 2 : + 1 nouvelle classe -> entraînement, évaluation sur les 3 classes vues
Tâche 3 : + 1 nouvelle classe -> évaluation sur les 4 classes vues
Tâche 4 : + 1 nouvelle classe -> évaluation sur les 5 classes
```

**Après chaque tâche**, l'évaluateur calcule :

1. l'accuracy sur toutes les classes vues (globale et par classe) ;
2. l'accuracy sur chaque tâche passée (matrice d'oubli) ;
3. les métriques **hors distribution**, de deux types :
   - décalage de conditions sur les mêmes classes (corruptions synthétiques, autre source d'images) ;
   - OOD sémantique (objets inconnus ; les classes non encore apprises comptent aussi) ;
4. la calibration (ECE) ;
5. les diagnostics géométriques (δ-hyperbolicité, courbure estimée, sondes par composante).

Le protocole est répété sur **plusieurs ordres de classes et plusieurs graines**. Les résultats sont rapportés avec intervalles de confiance.

## 3. Architecture

```
image ──► DINOv2-small (ViT-S/14, 10 premiers blocs) ──► tokens (384-d)
                                                          │
                                              ManifoldHead (projection)
                                                          ▼
                                      M = H^n × S^m × R^k   (tokens)
                                       │                        │
                              Predictor (JEPA)          Pooling tangent
                      contexte -> tokens masqués        (embedding global)
                          perte d_M²                            │
                                                          Classifier
                                                  (tête extensible / prototypes)
```

- **Tronc** : DINOv2-small, les deux derniers blocs sont retirés. Gel total ou partiel configurable. Si le tronc est gelé, les features du dernier bloc conservé peuvent être mises en cache.
- **ManifoldHead** : projette chaque token vers la variété produit (application exponentielle en 0 pour H, normalisation pour S, identité pour R). La signature `(n, m, k)` est une option de configuration.
- **Predictor** : prédit les représentations des régions masquées à partir du contexte visible ; perte = distance géodésique au carré de la variété produit.
- **Classifier** : opère sur l'embedding global, avec une tête qui s'étend à chaque nouvelle classe.

### Apprentissage continu

| Stratégie | Rôle |
|-----------|------|
| `finetune` | Borne basse (oubli maximal) |
| `joint` | Borne haute (toutes les données disponibles) |
| `replay` | Rejeu simple |
| `ewc` | Régularisation (Fisher diagonale empirique), référence de comparaison |
| `dual_memory` | **Méthode proposée** : mémoire rapide (rejeu récent) + mémoire lente (prototypes consolidés), transfert déclenché par un signal de surprise |

Le signal de surprise est l'erreur de prédiction normalisée du JEPA. Il déclenche la consolidation et sert aussi de score OOD.

## 4. Structure du dépôt

```
mccl/
├── configs/            # data/, model/, continual/, experiment/ (YAML + dataclasses)
├── data/               # raw/, splits/, ood/ (non versionné)
├── src/mccl/
│   ├── config.py       # dataclasses de configuration et validation
│   ├── data/           # splits, incremental, transforms, masking, ood
│   ├── geometry/       # manifolds, regularizers, diagnostics
│   ├── models/         # backbone, manifold_head, predictor, classifier, jepa
│   ├── continual/      # memory, surprise, consolidation, ewc, strategies
│   ├── training/       # losses, optim, trainer
│   ├── evaluation/     # metrics, continual_metrics, ood_metrics, calibration, evaluator
│   └── utils/          # logging, seed, checkpoint, viz
├── scripts/            # prepare_data, geometry_probe, run_incremental, analyze_results
├── tests/              # pytest
├── notebooks/          # analyses exploratoires
└── results/            # CSV/JSON par (stratégie, ordre, graine, tâche)
```

## 5. Installation

```bash
git clone <url-du-depot> mccl && cd mccl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Dépendances principales prévues : PyTorch, torchvision, NumPy, pandas, scikit-learn, PyYAML (ou Hydra), pytest. Les poids DINOv2-small sont chargés via `torch.hub` ou Hugging Face, selon l'accès réseau.

## 6. Utilisation (interfaces visées)

```bash
# 1. Préparer les données : dédoublonnage et splits figés
python scripts/prepare_data.py --config configs/data/waste5.yaml

# 2. Étape 0 : diagnostics géométriques sur features DINOv2, sans entraînement
python scripts/geometry_probe.py --config configs/data/waste5.yaml

# 3. Lancer une stratégie sur plusieurs ordres de classes et graines
python scripts/run_incremental.py \
    --config configs/experiment/dual_memory_mixed.yaml \
    --orders 5 --seeds 3

# 4. Agréger et visualiser les résultats
python scripts/analyze_results.py --results results/ --out reports/
```

## 7. Conventions de code

- Annotations de types sur toute l'API publique, `dataclass` pour les configurations et les sorties de modèles.
- Module `logging` (aucun `print`), niveau configurable, journaux par exécution.
- Docstrings complètes (rôle, arguments, retours, exceptions).
- Reproductibilité : graines fixées, déterminisme activé quand c'est possible, configuration et version du code sauvegardées avec chaque résultat.
- Tests (`pytest`) : opérations de variété (exp/log inverses, symétrie de la distance), absence de fuite entre splits, comportement des mémoires.

## 8. Risques identifiés

- **Instabilité hyperbolique** près du bord de la boule : projection, écrêtage des gradients, encodeur hybride euclidien-hyperbolique en repli.
- **Régularisation de type LeJEPA** non transposable à H et S : repli par l'espace tangent à l'origine.
- **Données peu hyperboliques** : l'étape 0 le révélera tôt ; le projet se replie alors sur un JEPA euclidien avec diagnostics.
- **Dérive des exemples mémorisés** si le tronc est dégelé : privilégier le rejeu en espace de features avec tronc gelé.

## 9. Feuille de route

- [x ] Données, splits anti-fuite, sonde linéaire de référence sur DINOv2 gelé
- [x ] Module `geometry/` et tests ; étape 0 (diagnostics sans entraînement)
- [x ] JEPA masqué et classifieur sur les 2 premières classes
- [x ] Boucle incrémentale : fine-tuning, joint, rejeu
- [ ] Mémoire duale, signal de surprise, EWC
- [ ] OOD, calibration, ablations (euclidien, H seul, S seul, mixte)

## 10. Références

- Gu, Sala, Gunel, Ré. *Learning mixed-curvature representations in product spaces*. ICLR 2019.
- Cetin, Chamberlain, Bronstein, Hunt. *Hyperbolic deep reinforcement learning*. ICLR 2023.
- Balestriero, LeCun. *LeJEPA: Provable and scalable self-supervised learning without the heuristics*. arXiv:2511.08544, 2025.
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks*. PNAS 114(13):3521–3526, 2017.
- McClelland, McNaughton, O'Reilly. *Why there are complementary learning systems in the hippocampus and neocortex*. Psychological Review 102(3):419–457, 1995.
- Benna, Fusi. *Computational principles of synaptic memory consolidation*. Nature Neuroscience 19:1697–1706, 2016.
- Knoblauch, Husain, Diethe. *Optimal continual learning has perfect memory and is NP-hard*. ICML 2020 (PMLR 119:5327–5337).

## 11. Auteur

Abdallah Houmadi SOIRFANE 