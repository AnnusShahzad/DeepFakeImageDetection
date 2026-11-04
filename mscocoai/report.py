"""Report artifact writers: confusion matrices, CSVs, sample grids, JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def save_confusion_png(cm: list[list[int]], labels: list[str], path: str | Path,
                       normalize: bool = False, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)
    fig_w = max(6, 1.5 + len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    im = ax.imshow(arr, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or ("Confusion matrix (row-normalized)" if normalize else "Confusion matrix (counts)"))
    fmt = ".2f" if normalize else ".0f"
    thresh = arr.max() / 2.0 if arr.size else 0.5
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center",
                    color="white" if arr[i, j] < thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_per_generator_csv(per_gen: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["generator", "accuracy", "precision", "recall", "f1", "auc", "pr_auc", "fpr_at_95tpr", "support"]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for gen, rep in per_gen.items():
            writer.writerow([gen, rep.get("accuracy"), rep.get("precision"), rep.get("recall"),
                             rep.get("f1"), rep.get("auc"), rep.get("pr_auc"),
                             rep.get("fpr_at_95tpr"), rep.get("support")])


def save_misclassified_csv(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "label_a", "label_b", "pred_a", "pred_b", "score_a", "score_b"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_samples_grid(ds, indices: list[int], preds: dict[int, dict], path: str | Path,
                      max_items: int = 24, cols: int = 6) -> None:
    """Grid of misclassified images with (true → pred) captions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idxs = list(indices)[:max_items]
    if not idxs:
        return
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes, idxs):
        img = ds.raw_image(int(idx))
        ax.imshow(img)
        p = preds.get(int(idx), {})
        ax.set_title(f"true={p.get('true')}\npred={p.get('pred')}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
