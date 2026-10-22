# mccl/src/mccl/continual/strategies.py
"""Stratégies d'apprentissage continu (baselines)."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from src.utils.logging import get_logger
from src.continual.ewc import EWC

log = get_logger(__name__)


class EWCMethod:
    """Enveloppe EWC pour l'entraînement continu."""
    
    def __init__(
        self, 
        model: nn.Module, 
        lam: float = 1e4, 
        n_samples: int = 512, 
        gamma: float = 1.0,
        **kwargs
    ) -> None:
        self._model = model
        self.ewc = EWC(model, lam=lam, n_samples=n_samples, gamma=gamma)
        
    def penalty(self) -> torch.Tensor:
        return self.ewc.penalty()

    def end_task(self, task: Any, train_loader: Any) -> None:
        """Appelée à la fin de chaque tâche pour mettre à jour la mémoire d'EWC."""
        task_idx = getattr(task, "index", "?")
        log.info("EWC : fin de la tâche %s, estimation de la Fisher...", task_idx)

        # 1. DÉTECTION DU NOMBRE TOTAL DE CLASSES VUES JUSQU'ICI
        # On suppose que 'task.classes' contient la liste des classes (ex: [10, 11, ..., 19])
        # ou que le modèle a déjà été étendu par le Trainer principal.
        # Si 'task' ne contient pas le total, on peut l'inférer à partir des classes de la tâche.
        new_classes = getattr(task, "classes", [])
        
        # Si votre framework stocke le nombre total de classes vues dans 'task', utilisez-le :
        # total_classes = getattr(task, "total_classes_seen", None)
        
        # Sinon, on calcule le nombre max de classes (en supposant que les labels sont globaux 0..N)
        if new_classes:
            total_classes = max(new_classes) + 1
        else:
            total_classes = None

        # 2. EXTENSION DU MODÈLE (CRUCIAL POUR ÉVITER L'ERREUR IndexError)
        if total_classes is not None:
            if hasattr(self._model, "expand_head"):
                self._model.expand_head(total_classes)
            elif hasattr(self._model, "update_output_layer"):
                self._model.update_output_layer(total_classes)
            elif hasattr(self._model, "head") and hasattr(self._model.head, "out_features"):
                # Fallback générique pour un simple nn.Linear
                current_out = self._model.head.out_features
                if current_out < total_classes:
                    log.info(f"EWC : Extension de la tête de {current_out} à {total_classes} classes.")
                    new_head = nn.Linear(self._model.head.in_features, total_classes).to(self._model.head.weight.device)
                    # On copie les anciens poids
                    with torch.no_grad():
                        new_head.weight[:current_out] = self._model.head.weight
                        new_head.bias[:current_out] = self._model.head.bias
                    self._model.head = new_head
            else:
                log.warning("EWC : Impossible d'étendre automatiquement la tête du modèle. L'erreur IndexError risque de se produire.")

        # 3. DÉFINITION DE LA LOSS POUR LA FISHER
        def loss_fn(logits, y):
            # Gestion des sorties de type JepaOutput
            logits = logits.logits if hasattr(logits, "logits") else logits
            
            # Sécurité : si le modèle n'a pas assez de sorties pour le label y, on ignore le batch
            # (Cela ne devrait plus arriver si l'étape 2 a bien fonctionné)
            if logits.shape[-1] <= y.max():
                log.warning(f"EWC : Le label max ({y.max()}) est hors des limites de sortie ({logits.shape[-1]}).")
                return torch.tensor(0.0, requires_grad=True, device=logits.device)
                
            return torch.nn.functional.cross_entropy(logits, y)

        # 4. ESTIMATION DE LA FISHER
        device = str(next(self._model.parameters()).device)
        try:
            self.ewc.estimate(loss_fn, train_loader, device=device)
            log.info("EWC : Fisher mise à jour sur la tâche %d", task.index if hasattr(task, 'index') else -1)
        except Exception as e:
            log.error(f"EWC : Échec de l'estimation de la Fisher : {e}")