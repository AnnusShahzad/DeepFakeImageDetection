"""Collecting results across run directories into CSV + markdown."""
from __future__ import annotations

import json

import pytest

from scripts import collect_results


def _run_dir(tmp_path, name: str, payload: dict) -> "Path":
    d = tmp_path / name
    d.mkdir()
    (d / "results.json").write_text(json.dumps(payload))
    return d


def test_collect_reads_results_json(tmp_path):
    run = _run_dir(tmp_path, "run1", {
        "run_name": "run1",
        "best_epoch": 2,
        "best_val": 0.3,
        "threshold": 0.42,
        "stop_reason": "completed",
        "config": {"task": "binary", "backbone": "resnet50", "freq_branch": "dct"},
        "epochs": [
            {"epoch": 1, "val_loss": 0.5, "val_metrics": {"accuracy": 0.7, "auc": 0.8}},
            {"epoch": 2, "val_loss": 0.3, "val_metrics": {"accuracy": 0.9, "auc": 0.95,
                                                           "balanced_accuracy": 0.88}},
        ],
    })
    rows = collect_results.collect([run])
    assert len(rows) == 1
    assert rows[0]["run"] == "run1"
    assert rows[0]["accuracy"] == pytest.approx(0.9)
    assert rows[0]["auc"] == pytest.approx(0.95)
    assert rows[0]["balanced_accuracy"] == pytest.approx(0.88)
    assert rows[0]["threshold"] == pytest.approx(0.42)


def test_collect_skips_dirs_without_results(tmp_path):
    (tmp_path / "empty").mkdir()
    assert collect_results.collect([tmp_path / "empty"]) == []


def test_write_csv_and_markdown(tmp_path):
    rows = [{"run": "a", "task": "binary", "accuracy": 0.9, "balanced_accuracy": 0.85}]
    collect_results.write_csv(rows, tmp_path / "r.csv")
    collect_results.write_markdown(rows, tmp_path / "r.md")
    assert "a" in (tmp_path / "r.csv").read_text()
    assert "| a |" in (tmp_path / "r.md").read_text()
