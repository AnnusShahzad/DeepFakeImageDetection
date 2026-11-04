import json

import pytest

from mscocoai.results import ResultsLog


def _epoch_row(epoch, val_loss=0.5, train_loss=1.0):
    return {
        "type": "epoch",
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": 1e-4,
        "is_best": True,
        "elapsed_sec": 1.0,
        "val_metrics": {"accuracy": 0.9, "macro_f1": 0.8},
    }


def test_epoch_row_is_upserted_not_duplicated(tmp_path):
    log = ResultsLog(tmp_path)
    log.append(_epoch_row(0, val_loss=0.5))
    log.append(_epoch_row(0, val_loss=0.4))
    assert len(log.epoch_rows()) == 1
    assert log.epoch_rows()[0]["val_loss"] == 0.4
    # the on-disk log is append-only but de-duplicated in memory
    assert len((tmp_path / "history.jsonl").read_text().strip().splitlines()) == 2


def test_truncated_final_line_is_dropped_and_earlier_rows_kept(tmp_path):
    (tmp_path / "history.jsonl").write_text(
        '{"type": "epoch", "epoch": 0, "val_loss": 0.5}\n'
        '{"type": "epoch", "epoch": 1, "val_loss": 0.4'
    )
    log = ResultsLog(tmp_path)
    assert [r["epoch"] for r in log.epoch_rows()] == [0]


def test_partial_rows_are_keyed_by_epoch_and_step(tmp_path):
    log = ResultsLog(tmp_path)
    log.append({"type": "partial", "epoch": 0, "step": 10, "train_loss": 0.7})
    log.append({"type": "partial", "epoch": 0, "step": 10, "train_loss": 0.6})
    assert len(log.rows()) == 1
    assert log.rows()[0]["train_loss"] == 0.6


def test_flush_writes_all_four_artifacts(tmp_path):
    log = ResultsLog(tmp_path)
    log.append(_epoch_row(0, val_loss=0.5))
    log.flush({"task": "binary"})
    for name in ("history.jsonl", "metrics.csv", "results.json", "summary.md"):
        assert (tmp_path / name).is_file()
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["best_val"] == pytest.approx(0.5) and res["best_epoch"] == 0
    assert "accuracy" in (tmp_path / "metrics.csv").read_text()
    assert "| 0 |" in (tmp_path / "summary.md").read_text()


def test_rows_includes_partial_and_epoch_rows(tmp_path):
    log = ResultsLog(tmp_path)
    log.append(_epoch_row(0))
    log.append({"type": "partial", "epoch": 1, "step": 5, "train_loss": 0.6})
    assert len(log.rows()) == 2
    assert len(log.epoch_rows()) == 1
    assert len(log.partial_rows()) == 1
    assert log.last_epoch() == 0
