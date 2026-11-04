"""Evaluation metrics (binary detection and 6-class attribution)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score,
)

from mscocoai.constants import LABEL_B_NAMES


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return float("nan")
    return float(fpr[idx].min())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(y_true)
    if len(classes) == 0:
        return 0.0
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in classes]))


def balanced_indices(y_true: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices of an equal-count, class-balanced subset (min class size)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    classes = np.unique(y_true)
    counts = [int((y_true == c).sum()) for c in classes]
    if not counts or min(counts) == 0:
        return np.array([], dtype=np.int64)
    take = min(counts)
    rng = np.random.default_rng(seed)
    idx: list[np.ndarray] = []
    for c in classes:
        members = np.flatnonzero(y_true == c)
        idx.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(idx))


def binary_report(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
        "positive_rate": float((y_true == 1).mean()),
        "balanced_accuracy": float(balanced_accuracy(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["fpr_at_95tpr"] = _fpr_at_tpr(y_true, y_score, 0.95)
    else:
        out["auc"] = out["pr_auc"] = out["fpr_at_95tpr"] = None
    return out


def attribution_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    per_class = {}
    for k, name in enumerate(LABEL_B_NAMES):
        mask = y_true == k
        support = int(mask.sum())
        y_true_bin = (y_true == k).astype(np.int64)
        y_pred_bin = (y_pred == k).astype(np.int64)
        if support:
            per_class[name] = {
                "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                "support": support,
            }
        else:
            per_class[name] = {"precision": None, "recall": None, "f1": None, "support": 0}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_B_NAMES))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_B_NAMES,
        "support": int(len(y_true)),
    }


def per_generator_detection(y_true, y_score, label_b, threshold: float = 0.5) -> dict:
    """Binary metrics on reals vs each generator separately (the paper's O5 analysis)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    label_b = np.asarray(label_b, dtype=np.int64)
    out: dict = {}
    real_mask = y_true == 0
    for gen_id, name in enumerate(LABEL_B_NAMES[1:], start=1):
        mask = real_mask | (label_b == gen_id)
        if mask.sum() == 0:
            continue
        out[name] = binary_report(y_true[mask], y_score[mask], threshold)
    return out


def balanced_binary_report(y_true, y_score, *, seed: int = 0, threshold: float = 0.5) -> dict:
    """Both raw-skew and class-balanced-subset binary reports.

    The balanced subset takes min(n_real, n_fake) samples per class (seeded).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    bi = balanced_indices(y_true, seed=seed)
    return {
        "balanced": binary_report(y_true[bi], y_score[bi], threshold),
        "raw": binary_report(y_true, y_score, threshold),
    }


def format_binary_table(report: dict) -> str:
    return (f"acc={report['accuracy']:.4f} f1={report['f1']:.4f} "
            f"auc={report.get('auc') if report.get('auc') is None else round(report['auc'], 4)} "
            f"pr_auc={report.get('pr_auc') if report.get('pr_auc') is None else round(report['pr_auc'], 4)} "
            f"fpr@95={report.get('fpr_at_95tpr') if report.get('fpr_at_95tpr') is None else round(report['fpr_at_95tpr'], 4)}")
