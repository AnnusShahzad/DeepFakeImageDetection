import json
import sys
from pathlib import Path

from mscocoai.engine import fit
from mscocoai.ensemble import main as ensemble_main
from tests.test_engine import _cfg


def test_ensemble_artifacts(tiny_dataset, tmp_path, monkeypatch):
    out = tmp_path / "run"
    fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
    report_dir = tmp_path / "ens"
    monkeypatch.setattr(sys, "argv", [
        "ensemble", "--task", "binary",
        "--checkpoints", str(out / "best.pt"), str(out / "last.pt"),
        "--split", "test", "--data-dir", str(tiny_dataset),
        "--img-size", "64", "--batch-size", "8", "--num-workers", "0",
        "--report-dir", str(report_dir),
    ])
    ensemble_main()
    report = json.loads((report_dir / "report.json").read_text())
    assert report["binary"]["support"] == 24
    assert len(report["weights"]) == 2
    assert (report_dir / "per_generator.csv").is_file()
