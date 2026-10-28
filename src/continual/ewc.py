# mccl/src/mccl/continual/ewc.py
"""Elastic Weight Consolidation (EWC) — baseline de comparaison."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.utils.logging import get_logger

log = get_logger(__name__)

LossFn = Callable[[Any, torch.Tensor], torch.Tensor]
Device = Union[str, torch.device]


def _overlap(p: torch.Tensor, ref: torch.Tensor) -> Tuple[slice, ...]:
    """Tranche de ``p`` couvrant la forme de ``ref`` (croissance en fin de dimension)."""
    if p.dim() != ref.dim() or any(r > s for r, s in zip(ref.shape, p.shape)):
        raise ValueError(f"Forme incompatible : paramètre {tuple(p.shape)}, référence {tuple(ref.shape)}")
    return tuple(slice(0, r) for r in ref.shape)


class EWC:
    """EWC (Kirkpatrick et al., 2017), Fisher diagonale empirique par échantillon.

    Fisher « empirique » : les gradients sont calculés avec les VRAIS labels
    (et non des labels échantillonnés depuis la prédiction du modèle). C'est
    l'approximation usuelle, mais elle diffère de la Fisher exacte.

    ``estimate`` doit être alimenté par des batches MÉLANGÉS : seuls les
    ``n_samples`` premiers échantillons sont utilisés, donc un loader trié par
    classe donnerait une Fisher biaisée vers les premières classes.
    """

    def __init__(
        self,
        model: nn.Module,
        lam: float = 1e4,
        n_samples: int = 512,
        gamma: float = 1.0,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples doit être >= 1")
        self.model = model
        self.lam = float(lam)
        self.n_samples = int(n_samples)
        self.gamma = float(gamma)

        self.fisher: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}
        self._n_estimated = 0
        self._warned_no_match = False

    # ------------------------------------------------------------------
    # Accumulation des gradients au carré
    # ------------------------------------------------------------------
    def _accumulate_loop(
        self,
        params: Dict[str, torch.nn.Parameter],
        acc: Dict[str, torch.Tensor],
        loss_fn: LossFn,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Un forward/backward par échantillon (référence, toujours valide)."""
        for i in range(x.shape[0]):
            self.model.zero_grad(set_to_none=True)
            loss = loss_fn(self.model(x[i : i + 1]), y[i : i + 1])
            loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    acc[n].addcmul_(p.grad, p.grad)

    def _accumulate_vmap(
        self,
        params: Dict[str, torch.nn.Parameter],
        acc: Dict[str, torch.Tensor],
        loss_fn: LossFn,
        x: torch.Tensor,
        y: torch.Tensor,
        chunk_size: int,
    ) -> None:
        """Gradients par échantillon vectorisés (torch.func), par chunks.

        Mémoire de pointe ≈ chunk_size × (nb de paramètres) × 4 octets.
        N'écrit dans ``acc`` qu'une fois TOUS les chunks réussis : en cas
        d'échec en cours de route, l'appelant peut rejouer en boucle sans
        double comptage.
        """
        from torch.func import functional_call, grad, vmap

        p_det = {n: p.detach() for n, p in params.items()}
        buffers = {n: b.detach() for n, b in self.model.named_buffers()}

        def loss_one(p: Dict[str, torch.Tensor], xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
            out = functional_call(self.model, (p, buffers), (xi.unsqueeze(0),))
            return loss_fn(out, yi.unsqueeze(0))

        per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))
        local = {n: torch.zeros_like(t) for n, t in acc.items()}
        for s in range(0, x.shape[0], chunk_size):
            g = per_sample(p_det, x[s : s + chunk_size], y[s : s + chunk_size])
            for n, gn in g.items():
                local[n] += (gn ** 2).sum(dim=0)
        for n in acc:
            acc[n] += local[n]

    # ------------------------------------------------------------------
    # Estimation de la Fisher
    # ------------------------------------------------------------------
    def estimate(
        self,
        loss_fn: LossFn,
        batches: Iterable,
        device: Optional[Device] = None,
        *,
        vectorized: bool = False,
        chunk_size: int = 8,
    ) -> None:
        """Estime la Fisher sur au plus ``n_samples`` échantillons et la fusionne.

        :param device: device des données ; par défaut, celui du modèle
        :param vectorized: si True, tente des gradients par échantillon via
            ``torch.func.vmap`` (bien plus rapide) ; repli automatique sur la
            boucle si ``loss_fn``/le modèle ne sont pas compatibles. À valider
            une fois contre la boucle (même Fisher, à la tolérance numérique).
        :param chunk_size: échantillons par passe vectorisée (borne la mémoire)
        """
        if device is None:
            device = next(self.model.parameters()).device

        was_training = self.model.training
        # Les gradients en cours (accumulation d'entraînement) sont préservés.
        saved_grads = {n: p.grad for n, p in self.model.named_parameters()}
        self.model.eval()

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        new_fisher = {n: torch.zeros_like(p) for n, p in params.items()}

        use_vmap = vectorized
        count = 0
        try:
            with torch.enable_grad():
                for batch in batches:
                    remaining = self.n_samples - count
                    if remaining <= 0:
                        break
                    x = batch[0][:remaining].to(device)
                    y = batch[1][:remaining].to(device)
                    if x.shape[0] == 0:
                        continue

                    if use_vmap:
                        try:
                            self._accumulate_vmap(params, new_fisher, loss_fn, x, y, chunk_size)
                        except Exception as exc:  # incompatibilité vmap : on rejoue en boucle
                            log.warning(
                                "EWC : Fisher vectorisée impossible (%s: %s) — repli sur la boucle.",
                                type(exc).__name__, exc,
                            )
                            use_vmap = False
                            self._accumulate_loop(params, new_fisher, loss_fn, x, y)
                    else:
                        self._accumulate_loop(params, new_fisher, loss_fn, x, y)
                    count += x.shape[0]
        finally:
            for n, p in self.model.named_parameters():
                p.grad = saved_grads[n]
            self.model.train(was_training)

        if count == 0:
            raise ValueError("EWC.estimate : aucun échantillon disponible")

        merged: Dict[str, torch.Tensor] = {}
        for n, f_new in new_fisher.items():
            f_new /= count
            old = self.fisher.get(n)
            if old is not None:
                padded = torch.zeros_like(f_new)
                padded[_overlap(f_new, old)] = old.to(f_new.device)
                f_new = self.gamma * padded + f_new
            merged[n] = f_new

        self.fisher = merged
        self.star = {n: params[n].detach().clone() for n in merged}
        self._n_estimated = count
        self._warned_no_match = False
        log.info(
            "EWC : Fisher estimée sur %d échantillons (%d tenseurs, %s)",
            count, len(merged), "vectorisée" if use_vmap else "boucle",
        )

    # ------------------------------------------------------------------
    # Pénalité
    # ------------------------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """Terme de pénalité EWC (0 si aucune Fisher n'a été estimée)."""
        total = None
        matched = 0
        for name, p in self.model.named_parameters():
            fisher = self.fisher.get(name)
            if fisher is None:
                continue
            star = self.star[name].to(p.device)
            sl = _overlap(p, star)
            term = (fisher.to(p.device) * (p[sl] - star) ** 2).sum()
            total = term if total is None else total + term
            matched += 1

        # Préfixes `module.` (DataParallel) / `_orig_mod.` (torch.compile) :
        # on signale aussi les correspondances PARTIELLES, pas seulement nulles.
        if self.fisher and matched < len(self.fisher) and not self._warned_no_match:
            log.warning(
                "EWC : %d/%d tenseurs de Fisher sans paramètre correspondant "
                "(noms modifiés ? DataParallel / torch.compile ?)",
                len(self.fisher) - matched, len(self.fisher),
            )
            self._warned_no_match = True

        if total is None:
            return torch.zeros((), device=next(self.model.parameters()).device)
        return 0.5 * self.lam * total

    # ------------------------------------------------------------------
    # Introspection / persistance
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "n_estimated": self._n_estimated,
            "lam": self.lam,
            "gamma": self.gamma,
            "n_params_tracked": len(self.fisher),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Fisher + ancre, sérialisables (checkpoint d'apprentissage continu).

        Les hyperparamètres (lam, gamma, n_samples) ne sont volontairement pas
        inclus : ils relèvent de la configuration, pas de l'état appris.
        """
        return {
            "fisher": {n: f.cpu() for n, f in self.fisher.items()},
            "star": {n: s.cpu() for n, s in self.star.items()},
            "n_estimated": self._n_estimated,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # On replace les tenseurs sur le device des paramètres : évite un
        # transfert hôte->GPU à chaque appel de penalty().
        dev = {n: p.device for n, p in self.model.named_parameters()}
        self.fisher = {n: f.to(dev.get(n, f.device)) for n, f in state["fisher"].items()}
        self.star = {n: s.to(dev.get(n, s.device)) for n, s in state["star"].items()}
        self._n_estimated = int(state["n_estimated"])
        self._warned_no_match = False