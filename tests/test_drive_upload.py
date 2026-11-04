import base64
import json
import urllib.request
from pathlib import Path

import pytest

import mscocoai.drive_upload as du


def test_chunk_plan_and_dry_run(tmp_path, capsys):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (5 * 1024 * 1024 + 3))  # > 5 MB
    du.upload_file(f, backend="apps_script", run_name="r",
                   upload_url="https://example.invalid/exec", token="t",
                   chunk_bytes=1 * 1024 * 1024, dry_run=True)
    out = capsys.readouterr().out
    assert "6 chunk(s)" in out
    assert not (tmp_path / "big.bin.manifest.json").exists()


def test_upload_requires_credentials(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"data")
    with pytest.raises(SystemExit):
        du.upload_file(f, backend="apps_script", run_name="r",
                       upload_url=None, token=None, dry_run=False)


def test_copy_backend_roundtrip(tmp_path):
    f = tmp_path / "w.pt"
    payload = b"weights-" * 1000
    f.write_bytes(payload)
    dest = du.upload_file(f, backend="copy", run_name="run1", drive_dir=str(tmp_path / "drive"))
    dest_path = Path(dest)
    assert dest_path.read_bytes() == payload
    manifest = du.list_remote(backend="copy", run_name="run1", drive_dir=str(tmp_path / "drive"))
    assert manifest["files"][0]["name"] == "w.pt"
    out = tmp_path / "dl" / "w.pt"
    du.download_file(backend="copy", run_name="run1", name="w.pt", out_path=out,
                     drive_dir=str(tmp_path / "drive"))
    assert out.read_bytes() == payload


def test_download_latest_apps_script(tmp_path, monkeypatch):
    payload = b"abc123" * 5000
    chunk = 1024
    parts = [payload[i:i + chunk] for i in range(0, len(payload), chunk)]
    calls = {"list": 0, "part": 0}

    def fake_post(url, payload_dict, timeout=300):
        action = payload_dict["action"]
        if action == "list":
            calls["list"] += 1
            return {"files": [{"name": "last_weights.pt",
                               "size": len(payload),
                               "parts": len(parts)}]}
        if action == "part":
            calls["part"] += 1
            idx = payload_dict["index"]
            return {"ok": True, "data": base64.b64encode(parts[idx]).decode()}
        if action == "purge":
            return {"ok": True}
        raise AssertionError(f"unexpected action {action}")

    monkeypatch.setattr(du, "_post_json", fake_post)
    out = tmp_path / "restore" / "last_weights.pt"
    du.download_file(backend="apps_script", run_name="myrun", name="last_weights.pt",
                     out_path=out, upload_url="https://example.invalid/exec", token="t")
    assert out.read_bytes() == payload
    assert calls == {"list": 1, "part": len(parts)}


def test_size_mismatch_deletes_tmp(tmp_path, monkeypatch):
    def fake_post(url, payload_dict, timeout=300):
        action = payload_dict["action"]
        if action == "list":
            return {"files": [{"name": "f.bin", "size": 9999, "parts": 1}]}
        if action == "part":
            return {"ok": True, "data": base64.b64encode(b"short").decode()}
        return {"ok": True}

    monkeypatch.setattr(du, "_post_json", fake_post)
    out = tmp_path / "f.bin"
    with pytest.raises(RuntimeError):
        du.download_file(backend="apps_script", run_name="r", name="f.bin",
                         out_path=out, upload_url="https://example.invalid/exec", token="t")
    assert not out.exists()


def test_upload_apps_script_calls_purge_and_chunks(tmp_path, monkeypatch, capsys):
    payload_file = tmp_path / "w.pt"
    payload_file.write_bytes(b"weights" * 100)
    actions = []

    def fake_post(url, payload_dict, timeout=300):
        actions.append(payload_dict["action"])
        if payload_dict["action"] == "store":
            assert payload_dict["data"]
        return {"ok": True}

    monkeypatch.setattr(du, "_post_json", fake_post)
    du.upload_file(payload_file, backend="apps_script", run_name="r",
                   upload_url="https://example.invalid/exec", token="t",
                   chunk_bytes=64)
    assert actions[0] == "purge"
    assert actions.count("store") == (payload_file.stat().st_size + 63) // 64
    assert actions[-1] == "finalize"
