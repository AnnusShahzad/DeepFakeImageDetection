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


def test_evaluate_with_tta_and_holdout(tiny_dataset, tmp_path, monkeypatch):
    ckpt = _train_tiny_binary(tiny_dataset, tmp_path)
    report_dir = tmp_path / "report_tta"
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(ckpt), "--split", "test",
        "--data-dir", str(tiny_dataset), "--img-size", "64",
        "--batch-size", "8", "--num-workers", "0",
        "--tta", "flip", "--holdout-generator", "sd3",
        "--report-dir", str(report_dir), "--no-samples",
    ])
    evaluate.main()
    report = json.loads((report_dir / "report.json").read_text())
    assert report["tta"] == "flip"
    assert report["holdout_generator"] == "sd3"
    # sd3 (4 images in the tiny test mirror) must be excluded
    assert report["binary"]["support"] == (4 * 6) - 4


def test_evaluate_inherits_resize_mode_from_checkpoint(tiny_dataset, tmp_path, monkeypatch):
    """A canonical-trained checkpoint must be evaluated with the canonical pipeline."""
    from mscocoai.train import TrainConfig
    from mscocoai.engine import fit
    from tests.test_engine import _cfg

    cfg = _cfg(tiny_dataset, tmp_path / "canon", epochs=1, resume="none")
    cfg.resize_mode = "canonical"
    fit(cfg)
    ckpt = tmp_path / "canon" / "best.pt"

    seen = {}
    real_build = evaluate.build_dataset

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(evaluate, "build_dataset", spy)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(ckpt), "--split", "test",
        "--data-dir", str(tiny_dataset), "--img-size", "64",
        "--batch-size", "8", "--num-workers", "0", "--no-samples",
        "--report-dir", str(tmp_path / "rep"),
    ])
    evaluate.main()
    assert seen["resize_mode"] == "canonical"
