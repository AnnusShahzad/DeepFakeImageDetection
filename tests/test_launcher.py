"""Shared-PC launcher: dry-run, execution, status persistence, skip-done."""
from __future__ import annotations

import json
import sys

import pytest

from mscocoai import launcher


def test_estimate_concurrency_scales_with_vram():
    assert launcher.estimate_concurrency(0.0) == 1
    assert launcher.estimate_concurrency(4.0) == 1
    assert launcher.estimate_concurrency(40.0) == 4


def test_load_jobs_rejects_bad_entry(tmp_path):
    bad = tmp_path / "jobs.json"
    bad.write_text(json.dumps([{"cmd": ["echo", "hi"]}]))
    with pytest.raises(SystemExit):
        launcher.load_jobs(bad)


def test_run_jobs_dry_run_does_not_execute(tmp_path):
    job = launcher.Job(name="j1", cmd=["echo", "hi"], out_dir=str(tmp_path / "o1"))
    n = launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                           status_path=str(tmp_path / "status.json"), dry_run=True)
    assert n == 1
    assert not (tmp_path / "o1").exists()


def test_run_jobs_executes_and_writes_status(tmp_path):
    job = launcher.Job(name="j1", cmd=[sys.executable, "-c", "print('ok-run')"],
                        out_dir=str(tmp_path / "o1"))
    launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                       status_path=str(tmp_path / "status.json"))
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["j1"]["status"] == "done"
    assert "ok-run" in (tmp_path / "logs" / "j1.log").read_text()


def test_run_jobs_skips_already_completed(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"j1": {"status": "done"}}))
    job = launcher.Job(name="j1", cmd=[sys.executable, "-c", "raise SystemExit(1)"],
                        out_dir=str(tmp_path / "o1"))
    n = launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                           status_path=str(status))
    assert n == 1
    assert json.loads(status.read_text())["j1"]["status"] == "done"


def test_failed_job_recorded_as_failed(tmp_path):
    job = launcher.Job(name="j2", cmd=[sys.executable, "-c", "raise SystemExit(3)"],
                        out_dir=str(tmp_path / "o2"))
    launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                       status_path=str(tmp_path / "status.json"))
    assert json.loads((tmp_path / "status.json").read_text())["j2"]["status"] == "failed"
