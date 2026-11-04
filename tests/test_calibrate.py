"""Calibration: threshold fit actually maximises the macro-F1 objective."""
from __future__ import annotations

import numpy as np
import pytest

from mscocoai import calibrate


def test_fit_threshold_picks_separable_boundary():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.1, 0.9, 1.0])
    thr = calibrate.fit_threshold(y, s, "macro_f1", grid=101)
    preds = (s >= thr).astype(int)
    assert (preds == y).mean() == 1.0


def test_fit_threshold_maximises_macro_f1():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    s = np.array([0.05, 0.3, 0.45, 0.4, 0.6, 0.75, 0.9, 0.98])
    thr = calibrate.fit_threshold(y, s, "macro_f1", grid=1001)
    assert 0.3 <= thr <= 0.9


def test_fit_threshold_unknown_objective_raises():
    y = np.array([0, 1])
    s = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        calibrate.fit_threshold(y, s, "banana")


def test_fit_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError):
        calibrate.fit_threshold(np.array([0, 1]), np.array([0.5]), "accuracy")
