# mccl/scripts/analyze_results.py
"""Analyse des résultats : tableaux, courbes, tests statistiques.

Parcourt results/**/*.json, agrège par ordre et par stratégie,
affiche un tableau récapitulatif, propose un test de Wilcoxon entre
deux groupes filtrés par regex sur le chemin.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_runs(results_dir: str) -> List[dict]:
    """Charge tous les JSON de runs, sauf geometry_probe."""
    runs = []
    for p in glob.glob(f"{results_dir}/**/*.json", recursive=True):
        if p.endswith("geometry_probe.json") or p.endswith("summary.json"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "avg_acc" not in d or "order" not in d:
                continue
            d["_path"] = p
            runs.append(d)
        except Exception as e:
            log.warning("JSON invalide %s : %s", p, e)
    return runs


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------
def summarize(runs: List[dict]) -> Dict[str, dict]:
    """Agrège avg_acc, forgetting, bwt par ordre de classes."""
    by_order: Dict[str, List[dict]] = {}
    for r in runs:
        key = str(tuple(r["order"]))
        by_order.setdefault(key, []).append(r)

    summary = {}
    for order, rs in by_order.items():
        accs = [r["avg_acc"] for r in rs]
        fgts = [r.get("forgetting", 0.0) for r in rs]
        bwts = [r.get("bwt", 0.0) for r in rs]
        summary[order] = {
            "n": len(rs),
            "avg_acc_mean": float(np.mean(accs)),
            "avg_acc_std": float(np.std(accs)),
            "forgetting_mean": float(np.mean(fgts)),
            "forgetting_std": float(np.std(fgts)),
            "bwt_mean": float(np.mean(bwts)),
            "bwt_std": float(np.std(bwts)),
        }
    return summary


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------
def wilcoxon(x: np.ndarray, y: np.ndarray):
    """Test de Wilcoxon signed-rank."""
    try:
        from scipy.stats import wilcoxon as _w
        stat, p = _w(x, y, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ImportError:
        log.warning("scipy indisponible : test Wilcoxon sauté")
        return float("nan"), float("nan")
    except Exception as e:
        log.warning("Wilcoxon a échoué : %s", e)
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse des résultats MCCL.")
    ap.add_argument("--dir", default="results")
    ap.add_argument("--group-a", default=None, help="regex sur le chemin")
    ap.add_argument("--group-b", default=None, help="regex sur le chemin")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    log.info("Runs trouvés : %d", len(runs))
    if not runs:
        return

    summary = summarize(runs)

    # --- Tableau ---
    print("\n=== Résumé par ordre de classes ===")
    header = f"{'ordre':>24} | {'n':>3} | {'avg_acc':>16} | {'forgetting':>16} | {'bwt':>16}"
    print(header)
    print("-" * len(header))
    for order, s in summary.items():
        print(f"{order:>24} | {s['n']:>3} | "
              f"{s['avg_acc_mean']:.4f} ± {s['avg_acc_std']:.4f} | "
              f"{s['forgetting_mean']:.4f} ± {s['forgetting_std']:.4f} | "
              f"{s['bwt_mean']:.4f} ± {s['bwt_std']:.4f}")

    # --- Wilcoxon optionnel ---
    if args.group_a and args.group_b:
        a = [r["avg_acc"] for r in runs if re.search(args.group_a, r["_path"])]
        b = [r["avg_acc"] for r in runs if re.search(args.group_b, r["_path"])]
        if a and b and len(a) == len(b):
            stat, p = wilcoxon(np.array(a), np.array(b))
            print(f"\nWilcoxon {args.group_a} vs {args.group_b}:")
            print(f"  stat = {stat:.4f}, p = {p:.4g}")
            summary["wilcoxon"] = {
                "a": args.group_a, "b": args.group_b,
                "stat": stat, "p": p,
            }
        else:
            log.warning("Wilcoxon : tailles différentes (%d vs %d)", len(a), len(b))

    # --- Écriture ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Résumé écrit : %s", out)


if __name__ == "__main__":
    main()