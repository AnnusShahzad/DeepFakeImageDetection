import json
from pathlib import Path

import pytest

from mscocoai.engine import fit


def _cfg(tiny_dataset, out_dir, epochs=1, resume="none"):
    from mscocoai.train import TrainConfig
    return TrainConfig(
        task="binary", backbone="resnet18", data_dir=str(tiny_dataset), limit=None,
        img_size=64, norm="imagenet", batch_size=8, eval_batch_size=8, accum_steps=1,
        epochs=epochs, lr=1e-3, weight_decay=1e-4, warmup_epochs=1, early_patience=5,
        num_workers=0, seed=42, out_dir=str(out_dir), resume=resume,
        balance_loss=False, balance_sampler=False, attr_class_weights=False,
        label_smoothing=0.0, ema_decay=0.0, save_steps=0,
        lambda_d=1.0, lambda_a=1.0, val_metrics_each_epoch=True,
        drive_backend="none", drive_dir=None, upload_every="never",
        apps_script_url=None, apps_script_token=None,
    )


def test_one_epoch_writes_artifacts(tiny_dataset, tmp_path):
    out = tmp_path / "run1"
    result = fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    assert (out / "best.pt").is_file()
    assert (out / "last.pt").is_file()
    assert (out / "LATEST").read_text().strip() == "last.pt"
    history = (out / "history.jsonl").read_text().strip().splitlines()
    assert len(history) == 1
    row = json.loads(history[0])
    for key in ("epoch", "train_loss", "val_loss", "lr", "is_best"):
        assert key in row
    assert result["best_val"] == pytest.approx(row["val_loss"], nan_ok=True)


def test_resume_continues_epoch(tiny_dataset, tmp_path):
    out = tmp_path / "run2"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
    rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1]


def test_resume_after_corrupt_last_pt(tiny_dataset, tmp_path):
    out = tmp_path / "run3"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    good_bytes = (out / "last.pt").read_bytes()
    (out / "last.pt.bak").write_bytes(good_bytes)
    (out / "last.pt").write_bytes(b"corrupted")
    # must fall back to last.pt.bak instead of crashing
    result = fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
    assert result["best_val"] is not None
    rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["epoch"] == 1


def test_history_has_one_row_per_epoch_after_resume(tiny_dataset, tmp_path):
    out = tmp_path / "run_rows"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
    rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1]
    assert all(r["type"] == "epoch" for r in rows)


def test_results_json_and_summary_exist(tiny_dataset, tmp_path):
    out = tmp_path / "run_json"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    res = json.loads((out / "results.json").read_text())
    assert res["best_epoch"] == 0 and res["best_val"] is not None
    assert (out / "summary.md").is_file() and (out / "metrics.csv").is_file()


def test_resume_after_truncated_history_line(tiny_dataset, tmp_path):
    out = tmp_path / "run_trunc"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    with (out / "history.jsonl").open("a") as fh:
        fh.write('{"type": "epoch", "epoch": 1, "val_loss": 0.1')
    fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
    rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1]


def test_upload_metrics_uploads_result_files(tiny_dataset, tmp_path, monkeypatch):
    out = tmp_path / "run_upload"
    cfg = _cfg(tiny_dataset, out, epochs=1, resume="none")
    cfg.drive_backend = "copy"
    cfg.drive_dir = str(tmp_path / "drive")
    cfg.upload_every = "epoch"
    cfg.upload_metrics = True
    seen = []
    monkeypatch.setattr("mscocoai.drive_upload.upload_files",
                        lambda paths, **kw: seen.extend(str(p) for p in paths) or [])
    fit(cfg)
    assert any(p.endswith("history.jsonl") for p in seen)
    assert any(p.endswith("results.json") for p in seen)


def test_max_hours_stops_cleanly(tiny_dataset, tmp_path):
    out = tmp_path / "run_budget"
    cfg = _cfg(tiny_dataset, out, epochs=10, resume="none")
    cfg.max_hours = 1e-9  # budget already exhausted
    result = fit(cfg)
    assert result["stop_reason"] == "time_budget"
    assert (out / "last.pt").is_file()


def test_multitask_detection_metrics_use_scores_not_hard_labels():
    """Regression: AUC/PR-AUC must be computed from detection *probabilities*.

    Feeding the 0/1 predictions into binary_report made AUC collapse onto
    balanced_accuracy (AUC of a thresholded score vector).
    """
    import torch

    from mscocoai.engine import evaluate_split

    class StubModel(torch.nn.Module):
        def forward(self, x):
            n = x.shape[0]
            det = torch.tensor([2.1972, 1.3863, -0.2007, -0.6190])[:n]
            attr = torch.zeros(n, 6)
            attr[:, 0] = 1.0
            return det, attr

    class StubLoader:
        def __iter__(self):
            yield {
                "image": torch.zeros(4, 3, 8, 8),
                "label_a": torch.tensor([1, 0, 1, 0]),
                "label_b": torch.tensor([0, 1, 0, 1]),
            }

    out = evaluate_split(StubModel(), StubLoader(), "cpu", "multitask")
    det = out["detection"]
    # probabilities [.900, .800, .450, .350] -> AUC 0.75
    # thresholded labels [1, 1, 0, 0]           -> AUC 0.50 (the old, wrong value)
    assert det["auc"] == pytest.approx(0.75, abs=1e-9)
    assert det["balanced_accuracy"] == pytest.approx(0.5, abs=1e-9)
