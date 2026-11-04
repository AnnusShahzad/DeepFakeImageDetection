"""CLI parser → TrainConfig round-trip; catches missing flags."""
from __future__ import annotations

import pytest

from mscocoai.train import TrainConfig, build_parser


def _parse(*args: str) -> TrainConfig:
    parsed = build_parser().parse_args(list(args))
    return TrainConfig(**{k.replace("-", "_"): v for k, v in vars(parsed).items()})


def test_all_default_args_produce_valid_config():
    cfg = _parse("--task", "binary", "--backbone", "resnet18", "--out-dir", "/tmp/run")
    assert cfg.lambda_d == 1.0 and cfg.lambda_a == 1.0
    assert cfg.val_metrics_each_epoch is False
    assert cfg.device == "auto" and cfg.data_layout == "auto"
    assert cfg.freq_branch == "none" and cfg.clip_init == "none"
    assert cfg.calibrate == "off" and cfg.tta == "none"


def test_all_tasks_parse():
    for task in ("binary", "attribution", "multitask"):
        cfg = _parse("--task", task, "--out-dir", "/tmp/run")
        assert cfg.task == task


def test_flag_roundtrip():
    cfg = _parse("--task", "multitask", "--out-dir", "/tmp/run",
                 "--freq-branch", "dct", "--freq-k", "16",
                 "--clip-init", "clip_vitl14", "--tta", "flip_scale",
                 "--calibrate", "macro_f1", "--holdout-generator", "sd3",
                 "--max-hours", "11", "--num-workers", "4",
                 "--val-metrics-each-epoch",
                 "--device", "cpu", "--data-layout", "parquet")
    assert cfg.freq_branch == "dct" and cfg.freq_k == 16
    assert cfg.clip_init == "clip_vitl14" and cfg.tta == "flip_scale"
    assert cfg.calibrate == "macro_f1" and cfg.holdout_generator == "sd3"
    assert cfg.max_hours == pytest.approx(11.0) and cfg.num_workers == "4"
    assert cfg.val_metrics_each_epoch is True
    assert cfg.device == "cpu" and cfg.data_layout == "parquet"
    # dict round-trips (config.json writes must never fail)
    import json
    json.dumps(cfg.as_dict(), default=str)
