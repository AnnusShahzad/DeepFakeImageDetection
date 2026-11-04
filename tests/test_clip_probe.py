import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mscocoai.clip_probe as clip_probe


def _write_fake_cache(out_dir: Path, n=240, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_b = rng.integers(0, 6, size=n)
    labels_a = (labels_b != 0).astype(np.int64)
    # well-separated synthetic embeddings per class
    means = rng.normal(0, 1, size=(6, dim)).astype(np.float32) * 6.0
    X = (means[labels_b] + rng.normal(0, 0.3, size=(n, dim))).astype(np.float32)
    np.save(out_dir / "embeddings_train.npy", X)
    np.save(out_dir / "label_a_train.npy", labels_a)
    np.save(out_dir / "label_b_train.npy", labels_b)
    np.save(out_dir / "embeddings_validation.npy", X[:60])
    np.save(out_dir / "label_a_validation.npy", labels_a[:60])
    np.save(out_dir / "label_b_validation.npy", labels_b[:60])
    np.save(out_dir / "embeddings_test.npy", X[:80])
    np.save(out_dir / "label_a_test.npy", labels_a[:80])
    np.save(out_dir / "label_b_test.npy", labels_b[:80])


def _args(out_dir: Path, extra=()):
    argv = [
        "--mode", "train", "--task", "binary", "--out-dir", str(out_dir),
        "--epochs", "200", "--early-patience", "25",
    ] + list(extra)
    parser = clip_probe.build_parser()
    return parser.parse_args(argv)


def test_probe_trains_and_eval_writes_report(tmp_path, monkeypatch, capsys):
    _write_fake_cache(tmp_path)
    args = _args(tmp_path)
    clip_probe._train_frozen_probe(args, tmp_path, "cpu")
    ckpt = tmp_path / "probe_binary.pt"
    assert ckpt.is_file()

    eval_argv = [
        "clip_probe", "--mode", "eval", "--task", "binary", "--out-dir", str(tmp_path),
        "--report-dir", str(tmp_path / "report"),
    ]
    monkeypatch.setattr(sys, "argv", eval_argv)
    clip_probe.main()
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["binary"]["support"] == 80
    assert report["binary"]["accuracy"] > 0.8
    assert (tmp_path / "report" / "per_generator.csv").is_file()
