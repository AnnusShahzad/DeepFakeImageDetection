import numpy as np
import pytest

from mscocoai import metrics
from mscocoai.metrics import (
    attribution_report, binary_report, per_generator_detection,
)
from mscocoai.constants import LABEL_B_NAMES


def test_binary_metrics_known_scores():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    rep = binary_report(y_true, y_score)
    assert rep["accuracy"] == pytest.approx(0.75)
    assert rep["precision"] == pytest.approx(1.0)
    assert rep["recall"] == pytest.approx(0.5)
    assert rep["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert rep["auc"] == pytest.approx(0.75)
    assert 0.0 <= rep["fpr_at_95tpr"] <= 1.0


def test_binary_report_keys():
    rep = binary_report(np.array([0, 1]), np.array([0.4, 0.6]))
    for key in ("accuracy", "precision", "recall", "f1", "auc", "pr_auc", "fpr_at_95tpr", "support"):
        assert key in rep


def test_macro_f1_known_and_labels_order():
    y_true = np.array([0, 1, 2, 3, 4, 5, 5])
    perfect = attribution_report(y_true, y_true)
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["labels"] == LABEL_B_NAMES
    cm = np.asarray(perfect["confusion_matrix"])
    assert cm.shape == (6, 6)
    assert np.allclose(cm, np.diag([1, 1, 1, 1, 1, 2]))
    worse = attribution_report(y_true, np.array([0, 1, 2, 3, 4, 4, 4]))
    assert worse["macro_f1"] < 1.0


def test_per_generator_detection_keys():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 30 + [1] * 150)
    y_score = rng.uniform(0, 1, size=180)
    label_b = np.array([0] * 30 + list(np.repeat([1, 2, 3, 4, 5], 30)))
    out = per_generator_detection(y_true, y_score, label_b)
    assert set(out) == set(LABEL_B_NAMES[1:])
    assert all("accuracy" in v for v in out.values())


def test_balanced_accuracy_ignores_class_frequency():
    y = np.array([0] * 90 + [1] * 10)
    pred = np.zeros(100, dtype=int)
    assert metrics.balanced_accuracy(y, pred) == pytest.approx(0.5)


def test_balanced_binary_report_reports_both_protocols():
    import numpy as np
    y = np.array([0] * 90 + [1] * 10)
    s = np.linspace(0.0, 1.0, 100)
    rep = metrics.balanced_binary_report(y, s, seed=0)
    assert {"balanced", "raw"} <= set(rep)
    assert rep["balanced"]["support"] == 20
    assert rep["raw"]["support"] == 100


def test_balanced_indices_are_balanced_and_seeded():
    import numpy as np
    y = np.array([0] * 50 + [1] * 10)
    idx = metrics.balanced_indices(y, seed=0)
    assert len(idx) == 20
    assert (y[idx] == 0).sum() == 10 and (y[idx] == 1).sum() == 10
