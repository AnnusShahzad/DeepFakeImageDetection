"""Preflight doctor: structure of the report and exit semantics."""
from __future__ import annotations

import pytest

from mscocoai import doctor


def test_collect_report_has_expected_sections(tmp_path):
    rep = doctor.collect_report(out_dir=str(tmp_path))
    for key in ("python", "torch", "cuda", "gpu", "ram_gb", "disk", "internet", "paths", "blockers"):
        assert key in rep
    assert isinstance(rep["ram_gb"], dict)
    assert {"total", "available"} <= set(rep["ram_gb"])
    assert {"root", "out_dir"} <= set(rep["disk"])
    assert rep["paths"]["out_dir_writable"] is True


def test_render_returns_text(tmp_path):
    rep = doctor.collect_report(out_dir=str(tmp_path))
    text = doctor.render(rep)
    assert isinstance(text, str)
    assert "python" in text


def test_low_disk_records_blocker(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_free_disk", lambda path: 1.0)
    rep = doctor.collect_report(out_dir=str(tmp_path))
    assert any("free disk" in b for b in rep["blockers"])


def test_unwritable_out_dir_flagged(tmp_path):
    target = tmp_path / "nope" / "deeper"
    # parent "nope" exists as a file, so creating "deeper" inside it must fail
    (tmp_path / "nope").write_text("i am a file, not a directory")
    rep = doctor.collect_report(out_dir=str(target))
    assert rep["paths"]["out_dir_writable"] is False


def test_expected_writes_into_out_dir(tmp_path):
    out = tmp_path / "reused"
    doctor.collect_report(out_dir=str(out))
    assert not (out / ".doctor_probe").exists()  # probe is cleaned up
    assert out.is_dir()
