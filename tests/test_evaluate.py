import json
import sys
from pathlib import Path

import numpy as np
import pytest

import mscocoai.evaluate as evaluate
from mscocoai.engine import fit
from tests.test_engine import _cfg


def _train_tiny_binary(tiny_dataset, tmp_path):
    out = tmp_path / "run"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    return out / "best.pt"


def test_evaluate_writes_artifacts(tiny_dataset, tmp_path, monkeypatch):
    ckpt = _train_tiny_binary(tiny_dataset, tmp_path)
    report_dir = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(ckpt), "--split", "test",
        "--data-dir", str(tiny_dataset), "--img-size", "64",
        "--batch-size", "8", "--num-workers", "0",
        "--report-dir", str(report_dir), "--no-samples",
    ])
    evaluate.main()
    report = json.loads((report_dir / "report.json").read_text())
    assert report["binary"]["support"] == 4 * 6
    assert 0.0 <= report["binary"]["accuracy"] <= 1.0
    assert (report_dir / "per_generator.csv").is_file()
    assert (report_dir / "misclassified.csv").is_file()


def test_report_artifacts(tiny_dataset, tmp_path):
    from mscocoai import report as rep
    from mscocoai.data import LocalFolderDataset
    cm = [[3, 1, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0],
          [0, 0, 1, 3, 0, 0], [0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 4]]
    labels = ["real", "sd21", "sdxl", "sd3", "dalle3", "mj6"]
    rep.save_confusion_png(cm, labels, tmp_path / "cm.png")
    rep.save_confusion_png(cm, labels, tmp_path / "cm_norm.png", normalize=True)
    rep.save_per_generator_csv({"sd3": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9,
                                        "f1": 0.9, "auc": 0.95, "pr_auc": 0.9,
                                        "fpr_at_95tpr": 0.1, "support": 10}},
                               tmp_path / "pg.csv")
    ds = LocalFolderDataset(tiny_dataset, "test")
    rep.save_samples_grid(ds, [0, 1], {0: {"true": 0, "pred": 1}, 1: {"true": 1, "pred": 1}},
                          tmp_path / "grid.png")
    rep.save_misclassified_csv([{"index": 0, "label_a": 0, "pred_a": 1, "label_b": 0,
                                "pred_b": None, "score_a": 0.6, "score_b": None}],
                               tmp_path / "mis.csv")
    for name in ("cm.png", "cm_norm.png", "pg.csv", "grid.png", "mis.csv"):
        assert (tmp_path / name).is_file() and (tmp_path / name).stat().st_size > 0
