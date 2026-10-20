# mccl/src/mccl/utils/viz.py
"""Visualisations ultra-pro pour MCCL."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence


PALETTE = {
    "finetune": "#d62728",
    "joint": "#2ca02c",
    "replay": "#1f77b4",
    "ewc": "#ff7f0e",
    "dual_memory": "#9467bd",
    "euclid": "#8c564b",
    "hyper_only": "#e377c2",
    "sphere_only": "#17becf",
    "mixed": "#bcbd22",
}

DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_for(name: str, idx: int = 0) -> str:
    return PALETTE.get(name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


def _get_cmap(name: str):
    """Compat : récupère une colormap sans l'API dépréciée plt.cm.get_cmap."""
    import matplotlib.pyplot as plt
    return plt.get_cmap(name).copy()


def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------------------
# 1. Matrice d'accuracy
# ---------------------------------------------------------------------------
def plot_accuracy_matrix(
    acc_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Matrice d'accuracy (tâche × classe)",
    save_path: Optional[str] = None,
    cmap: str = "YlGnBu",
    figsize: Optional[tuple] = None,
    annotate: bool = True,
):
    import matplotlib.pyplot as plt
    _setup_style()

    T, C = acc_matrix.shape
    if figsize is None:
        figsize = (1.2 + 0.7 * C, 1.0 + 0.55 * T)

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(acc_matrix)
    cmap_obj = _get_cmap(cmap)
    cmap_obj.set_bad(color="#eeeeee")

    im = ax.imshow(masked, cmap=cmap_obj, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xlabel("Classe évaluée")
    ax.set_ylabel("Après tâche")

    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"T{i}" for i in range(T)])

    if annotate:
        for i in range(T):
            for j in range(C):
                v = acc_matrix[i, j]
                if not np.isnan(v):
                    color = "white" if v > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center",
                            color=color, fontsize=8, fontweight="bold")

    for i in range(1, T):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)

    ax.set_title(title, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Accuracy")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Courbes d'apprentissage
# ---------------------------------------------------------------------------
def plot_learning_curves(
    history: List[Dict],
    title: str = "Courbes d'apprentissage",
    save_path: Optional[str] = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt
    _setup_style()

    if not history:
        return

    epochs = [h.get("epoch", i) for i, h in enumerate(history)]
    keys = [k for k in history[0].keys() if k != "epoch"]

    n_plots = len(keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.6))
    if n_plots == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        vals = [h.get(k, np.nan) for h in history]
        color = "#d62728" if "loss" in k else "#2ca02c"
        ax.plot(epochs, vals, marker="o", markersize=4,
                linewidth=2, color=color)
        ax.fill_between(epochs, vals, alpha=0.15, color=color)
        ax.set_xlabel("Époque")
        ax.set_ylabel(k)
        ax.set_title(k)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# 3. Comparaison de stratégies
# ---------------------------------------------------------------------------
def plot_strategy_comparison(
    results: Dict[str, np.ndarray],
    metric: str = "avg_acc",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    names = list(results.keys())
    colors = [_color_for(n, i) for i, n in enumerate(names)]

    metrics = {}
    for name, mat in results.items():
        metrics[name] = {
            "avg_acc": float(np.nanmean(mat[-1])),
            "forgetting": _compute_forgetting(mat),
            "bwt": _compute_bwt(mat),
        }

    if metric == "per_class":
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(class_names or [])),
                                       4.5))
        last_row = {n: mat[-1] for n, mat in results.items()}
        n_classes = len(next(iter(last_row.values())))
        x = np.arange(n_classes)
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [last_row[name][c] if not np.isnan(last_row[name][c])
                    else 0 for c in range(n_classes)]
            ax.bar(x + i * width - 0.4 + width / 2, vals,
                   width=width, label=name, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(class_names or [f"c{c}" for c in range(n_classes)])
        ax.set_ylabel("Accuracy finale")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best")
        ax.set_title(title or "Accuracy finale par classe")
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vals = [metrics[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} par stratégie")
        if metric in ("avg_acc", "bwt"):
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Courbes de rétention
# ---------------------------------------------------------------------------
def plot_retention_curves(
    results: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, strat in zip(axes, names):
        mat = results[strat]
        T, C = mat.shape
        x = np.arange(T)
        for c in range(C):
            col = mat[:, c]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                continue
            color = DEFAULT_COLORS[c % len(DEFAULT_COLORS)]
            label = class_names[c] if class_names else f"classe {c}"
            ax.plot(x[valid], col[valid], marker="o", markersize=4,
                    linewidth=2, color=color, label=label)
        ax.set_title(strat)
        ax.set_xlabel("Après tâche")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{i}" for i in x])
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Rétention par classe au fil des tâches",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Radar multi-critères
# ---------------------------------------------------------------------------
def plot_radar(
    results: Dict[str, np.ndarray],
    criteria: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    names = list(results.keys())
    metrics_per_strat = {}
    for name, mat in results.items():
        avg = float(np.nanmean(mat[-1]))
        forget = _compute_forgetting(mat)
        bwt = _compute_bwt(mat)
        metrics_per_strat[name] = {
            "avg_acc": avg,
            "1 - forgetting": max(0.0, 1.0 - forget),
            "bwt normalisé": max(0.0, min(1.0, (bwt + 0.5) / 1.0)),
            "stabilité": max(0.0, min(1.0, 1.0 - forget)),
            "plasticité": avg,
        }

    if criteria is None:
        criteria = list(next(iter(metrics_per_strat.values())).keys())

    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    for i, name in enumerate(names):
        values = [metrics_per_strat[name][c] for c in criteria]
        values += values[:1]
        color = _color_for(name, i)
        ax.plot(angles, values, linewidth=2, color=color, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Comparaison multi-critères", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Diagnostics géométriques (courbures)
# ---------------------------------------------------------------------------
def plot_geometry_diagnostics(
    diagnostics_per_task: List[Dict],
    title: str = "Diagnostics géométriques par tâche",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    if not diagnostics_per_task:
        return

    keys = [k for k in diagnostics_per_task[0].keys() if k != "task"]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        vals = [d.get(k, np.nan) for d in diagnostics_per_task]
        tasks = list(range(len(vals)))
        ax.plot(tasks, vals, marker="o", markersize=6,
                linewidth=2, color="#9467bd")
        ax.fill_between(tasks, vals, alpha=0.15, color="#9467bd")
        ax.set_xlabel("Tâche")
        ax.set_ylabel(k)
        ax.set_title(k)
        ax.set_xticks(tasks)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 7. Embeddings 2D
# ---------------------------------------------------------------------------
def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    class_names: Optional[List[str]] = None,
    title: str = "Embeddings 2D",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    if method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30,
                    random_state=0, init="pca").fit_transform(embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
            proj = umap.UMAP(n_components=2,
                             random_state=0).fit_transform(embeddings)
        except ImportError:
            raise ImportError("pip install umap-learn")
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = np.unique(labels)
    for i, c in enumerate(unique_labels):
        m = labels == c
        name = class_names[c] if class_names else f"classe {c}"
        ax.scatter(proj[m, 0], proj[m, 1], s=10, alpha=0.7,
                   color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                   label=name, edgecolors="none")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 8. Tableau récapitulatif
# ---------------------------------------------------------------------------
def summary_table(
    results: Dict[str, np.ndarray],
    save_csv: Optional[str] = None,
) -> str:
    rows = []
    for name, mat in results.items():
        avg = float(np.nanmean(mat[-1]))
        forget = _compute_forgetting(mat)
        bwt = _compute_bwt(mat)
        last = mat[-1]
        valid = ~np.isnan(last)
        if valid.sum() == 0:
            continue
        best_c = int(np.nanargmax(last))
        worst_c = int(np.nanargmin(last))
        rows.append({
            "stratégie": name,
            "avg_acc": round(avg, 4),
            "forgetting": round(forget, 4),
            "bwt": round(bwt, 4),
            "best_class": best_c,
            "worst_class": worst_c,
        })

    if not rows:
        return ""
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows))
                  for h in headers}

    def fmt_row(row):
        return " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)

    sep = "-+-".join("-" * col_widths[h] for h in headers)
    table = "\n".join([
        fmt_row({h: h for h in headers}),
        sep,
        *[fmt_row(r) for r in rows],
    ])

    if save_csv:
        import csv
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)

    return table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_forgetting(acc_matrix: np.ndarray) -> float:
    T, C = acc_matrix.shape
    if T < 2:
        return 0.0
    drops = []
    for c in range(C):
        col = acc_matrix[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        best = np.nanmax(col)
        final = col[-1]
        if not np.isnan(final):
            drops.append(best - final)
    return float(np.mean(drops)) if drops else 0.0


def _compute_bwt(acc_matrix: np.ndarray) -> float:
    T, C = acc_matrix.shape
    if T < 2:
        return 0.0
    bwt = []
    for c in range(min(C, T)):
        col = acc_matrix[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        first = col[np.argmax(valid)]
        final = col[-1]
        if not np.isnan(first) and not np.isnan(final):
            bwt.append(final - first)
    return float(np.mean(bwt)) if bwt else 0.0


# ---------------------------------------------------------------------------
# Dashboard complet
# ---------------------------------------------------------------------------
def plot_dashboard(
    results: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    history_per_strategy: Optional[Dict[str, List[Dict]]] = None,
    diagnostics_per_task: Optional[List[Dict]] = None,
    save_dir: Optional[str] = None,
):
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("\n=== Moyennes ===")
    print(summary_table(
        results,
        save_csv=f"{save_dir}/summary.csv" if save_dir else None,
    ))

    for name, mat in results.items():
        path = f"{save_dir}/acc_matrix_{name}.png" if save_dir else None
        plot_accuracy_matrix(mat, class_names=class_names,
                             title=f"Matrice d'accuracy — {name}",
                             save_path=path)

    plot_retention_curves(
        results, class_names=class_names,
        save_path=f"{save_dir}/retention.png" if save_dir else None,
    )

    for metric in ["avg_acc", "forgetting", "bwt"]:
        plot_strategy_comparison(
            results, metric=metric,
            save_path=f"{save_dir}/compare_{metric}.png" if save_dir else None,
        )

    plot_radar(
        results,
        save_path=f"{save_dir}/radar.png" if save_dir else None,
    )

    if history_per_strategy:
        for name, hist in history_per_strategy.items():
            plot_learning_curves(
                hist, title=f"Courbes d'apprentissage — {name}",
                save_path=f"{save_dir}/curves_{name}.png" if save_dir else None,
            )

    if diagnostics_per_task:
        plot_geometry_diagnostics(
            diagnostics_per_task,
            save_path=f"{save_dir}/geometry.png" if save_dir else None,
        )


# ---------------------------------------------------------------------------
# Courbures
# ---------------------------------------------------------------------------
def plot_curvature_evolution(
    diagnostics_per_task: List[Dict],
    title: str = "Évolution des courbures au fil des tâches",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    if not diagnostics_per_task:
        return

    keys = [k for k in diagnostics_per_task[0].keys() if k != "task"]

    n = len(keys)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    tasks = [d.get("task", i) for i, d in enumerate(diagnostics_per_task)]

    for ax, k in zip(axes, keys):
        vals = [d.get(k, np.nan) for d in diagnostics_per_task]
        color = _color_for_key(k)
        ax.plot(tasks, vals, marker="o", markersize=8, linewidth=2.5,
                color=color, markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(tasks, vals, alpha=0.15, color=color)
        ax.set_xlabel("Tâche")
        ax.set_ylabel(k)
        ax.set_title(k, fontsize=11)
        ax.set_xticks(tasks)
        ax.grid(alpha=0.3)

        if vals and not np.isnan(vals[-1]):
            ax.annotate(
                f"{vals[-1]:.3f}",
                xy=(tasks[-1], vals[-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color=color,
            )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def _color_for_key(k: str) -> str:
    mapping = {
        "delta_hyperbolicity": "#d62728",
        "curvature_probe": "#ff7f0e",
        "curvature_toponogov": "#2ca02c",
        "mean_curvature_local": "#9467bd",
        "std_curvature_local": "#8c564b",
        "median_curvature_local": "#e377c2",
        "mean_norm": "#1f77b4",
        "max_norm": "#17becf",
        "std_norm": "#bcbd22",
    }
    return mapping.get(k, "#7f7f7f")


def plot_curvature_comparison(
    results_per_strategy: Dict[str, List[Dict]],
    metric: str = "delta_hyperbolicity",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (name, diagnostics) in enumerate(results_per_strategy.items()):
        tasks = [d.get("task", j) for j, d in enumerate(diagnostics)]
        vals = [d.get(metric, np.nan) for d in diagnostics]
        color = _color_for(name, i)
        ax.plot(tasks, vals, marker="o", markersize=6, linewidth=2,
                color=color, label=name)
        ax.fill_between(tasks, vals, alpha=0.1, color=color)

    ax.set_xlabel("Tâche")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Évolution de {metric} par stratégie")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_curvature_heatmap(
    diagnostics_per_task: List[Dict],
    metrics: Optional[List[str]] = None,
    title: str = "Diagnostics géométriques par tâche",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    _setup_style()

    if not diagnostics_per_task:
        return

    if metrics is None:
        metrics = [k for k in diagnostics_per_task[0].keys() if k != "task"]

    n_tasks = len(diagnostics_per_task)
    mat = np.full((len(metrics), n_tasks), np.nan)
    for j, d in enumerate(diagnostics_per_task):
        for i, k in enumerate(metrics):
            if k in d:
                mat[i, j] = d[k]

    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * n_tasks, 0.6 * len(metrics) + 1.5))

    norm = np.zeros_like(mat)
    for i in range(len(metrics)):
        row = mat[i]
        if np.all(np.isnan(row)):
            continue
        rmin, rmax = np.nanmin(row), np.nanmax(row)
        if rmax - rmin > 1e-9:
            norm[i] = (row - rmin) / (rmax - rmin)
        else:
            norm[i] = 0.5

    masked = np.ma.masked_invalid(norm)
    cmap = _get_cmap("RdYlGn")
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f"T{d.get('task', j)}"
                        for j, d in enumerate(diagnostics_per_task)])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Tâche")

    for i in range(len(metrics)):
        for j in range(n_tasks):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="black", fontsize=7)

    ax.set_title(title, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="Score normalisé (rouge=bas, vert=haut)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()