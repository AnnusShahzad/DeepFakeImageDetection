"""Fit a decision threshold on validation scores (never used on test)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def fit_threshold(y_true: np.ndarray, scores: np.ndarray,
                  objective: str = "macro_f1", grid: int = 1001) -> float:
    """Return the threshold in (0, 1) maximising the objective.

    Ties are broken towards the smallest threshold (more sensitive detector).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f"y_true and scores shapes differ: {y_true.shape} vs {scores.shape}")
    objective = (objective or "macro_f1").lower()
    if objective not in ("macro_f1", "f1", "accuracy"):
        raise ValueError(f"unknown objective: {objective}")

    candidates = np.linspace(0.0, 1.0, int(grid))
    best_value, best_thr = float("-inf"), 0.5
    for thr in candidates:
        preds = (scores >= thr).astype(np.int64)
        if objective == "accuracy":
            value = accuracy_score(y_true, preds)
        else:  # macro_f1 or f1
            value = f1_score(y_true, preds, average="macro", zero_division=0)
        if value > best_value:
            best_value, best_thr = float(value), float(thr)
    return best_thr
