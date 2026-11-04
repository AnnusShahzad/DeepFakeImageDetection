"""Upload / download checkpoints to Google Drive — no admin access required.

Backends:
  copy        atomic copy into a folder (use a Google Drive for Desktop synced
              folder on the lab PC -> files sync to Drive automatically)
  apps_script POST base64 chunks to a tiny Google Apps Script web app that runs
              as *your* Google account (see apps_script/uploader.gs). No Cloud
              project, no OAuth client, no service account, no API key.
  (kaggle)    do nothing here — rely on Kaggle "Save Version" output persistence.

Env vars (fallbacks for CLI flags):
  MSCOCOAI_APPS_SCRIPT_URL     web app URL printed after deploying uploader.gs
  MSCOCOAI_APPS_SCRIPT_TOKEN   the shared secret you set in uploader.gs
  MSCOCOAI_DRIVE_DIR           destination folder for --backend copy

Files > --chunk-mb are uploaded as .part-0000… parts and reassembled on
download-latest (client-side concat), so files of any size work.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK_BYTES = 24 * 1024 * 1024  # stay comfortably under Apps Script's 50 MB POST limit


# ------------------------------------------------------------- low-level HTTP
def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — user-supplied endpoint
        body = resp.read().decode("utf-8", "replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Apps Script returned non-JSON response ({exc}): {body[:200]}") from exc


def _retry(fn, attempts: int = 3, base_delay: float = 2.0):
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    assert last_exc is not None
    raise last_exc


def _require_apps_script(url: str | None, token: str | None) -> tuple[str, str]:
    url = url or os.environ.get("MSCOCOAI_APPS_SCRIPT_URL")
    token = token or os.environ.get("MSCOCOAI_APPS_SCRIPT_TOKEN")
    if not url or not token:
        raise SystemExit(
            "Apps Script backend needs --apps-script-url and --apps-script-token\n"
            "(or env MSCOCOAI_APPS_SCRIPT_URL / MSCOCOAI_APPS_SCRIPT_TOKEN).\n"
            "See apps_script/uploader.gs for the 5-minute setup."
        )
    return url, token


# ------------------------------------------------------------------- backends
def upload_file(path: str | os.PathLike, *, backend: str, run_name: str,
                drive_dir: str | None = None, download_from_name: str | None = None,
                upload_url: str | None = None, token: str | None = None,
                chunk_bytes: int = CHUNK_BYTES, dry_run: bool = False) -> str | None:
    """Upload one file. Returns destination description (or None on dry-run)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    parts = max(1, (size + chunk_bytes - 1) // chunk_bytes)

    if backend == "none":
        print(f"[drive_upload] backend=none — skipping upload of {path.name}")
        return None

    if backend == "copy":
        if not drive_dir:
            if dry_run:
                print(f"[drive_upload] dry-run: would copy {path} -> <drive-dir>/{run_name}/")
                return None
            raise SystemExit("--drive-dir is required for --backend copy "
                             "(or set MSCOCOAI_DRIVE_DIR)")
        dest_dir = Path(drive_dir) / run_name
        dest = dest_dir / path.name
        if dry_run:
            print(f"[drive_upload] dry-run: would copy {path} -> {dest}")
            return str(dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with path.open("rb") as src, tmp.open("wb") as dst:
            while True:
                buf = src.read(8 << 20)
                if not buf:
                    break
                dst.write(buf)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, dest)
        print(f"[drive_upload] copied -> {dest}")
        return str(dest)

    if backend == "apps_script":
        url, token = _require_apps_script(upload_url, token)
        if dry_run:
            print(f"[drive_upload] dry-run: would upload {path} ({size} bytes) as {parts} chunk(s) to {url} (run={run_name})")
            return None
        purge = _retry(lambda: _post_json(url, {"token": token, "action": "purge",
                                                "run": run_name, "name": path.name}))
        if not purge.get("ok"):
            raise RuntimeError(f"purge of old parts failed: {purge}")
        with path.open("rb") as fh:
            for index in range(parts):
                chunk = fh.read(chunk_bytes)
                payload = {
                    "token": token,
                    "action": "store",
                    "run": run_name,
                    "name": path.name,
                    "index": index,
                    "total": parts,
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
                resp = _retry(lambda: _post_json(url, payload))
                if not resp.get("ok"):
                    raise RuntimeError(f"chunk {index} rejected by Apps Script: {resp}")
                print(f"[drive_upload] chunk {index + 1}/{parts} of {path.name} stored")
        resp = _retry(lambda: _post_json(url, {
            "token": token, "action": "finalize",
            "run": run_name, "name": path.name, "total": parts,
        }))
        if not resp.get("ok"):
            raise RuntimeError(f"finalize rejected: {resp}")
        print(f"[drive_upload] uploaded {path.name} -> Drive folder MscocoAI-Weights/{run_name}/")
        return f"drive:{run_name}/{path.name}"

    raise SystemExit(f"Unknown backend: {backend}")


def upload_files(paths, *, backend: str, run_name: str,
                 drive_dir: str | None = None, upload_url: str | None = None,
                 token: str | None = None, chunk_bytes: int = CHUNK_BYTES,
                 dry_run: bool = False) -> list[str]:
    """Upload several files, tolerating individual failures (never raises).

    Returns the list of files that were (or, in dry-run, would be) uploaded.
    """
    done: list[str] = []
    base = Path(drive_dir) if drive_dir else None
    for file_path in paths:
        path = Path(file_path)
        try:
            if not path.is_file():
                print(f"[drive_upload] skipping missing file: {path}")
                continue
            if backend == "copy" and base is not None:
                dest = base / run_name / path.name
                if dest.is_file() and dest.stat().st_size == path.stat().st_size:
                    print(f"[drive_upload] {path.name} unchanged on Drive — skipping")
                    done.append(str(dest))
                    continue
            out = upload_file(path, backend=backend, run_name=run_name,
                              drive_dir=drive_dir, upload_url=upload_url,
                              token=token, chunk_bytes=chunk_bytes, dry_run=dry_run)
            done.append(out or str(path))
        except Exception as exc:  # noqa: BLE001 — one file failing must not stop the rest
            print(f"[drive_upload] WARNING: upload of {path.name} failed ({exc})")
    return done


def list_remote(*, backend: str, run_name: str, upload_url: str | None = None,
                token: str | None = None, drive_dir: str | None = None) -> dict:
    if backend == "apps_script":
        url, token = _require_apps_script(upload_url, token)
        return _retry(lambda: _post_json(url, {"token": token, "action": "list", "run": run_name}))
    if backend == "copy":
        base = Path(drive_dir or os.environ.get("MSCOCOAI_DRIVE_DIR") or ".") / run_name
        files = []
        if base.is_dir():
            for p in sorted(base.iterdir()):
                if p.is_file():
                    files.append({"name": p.name, "size": p.stat().st_size, "parts": 1})
        return {"files": files}
    raise SystemExit(f"Unknown backend: {backend}")


def download_file(*, backend: str, run_name: str, name: str, out_path: str | os.PathLike,
                  drive_dir: str | None = None, upload_url: str | None = None,
                  token: str | None = None, dry_run: bool = False) -> str:
    out_path = Path(out_path)
    if backend == "copy":
        src = Path(drive_dir or os.environ.get("MSCOCOAI_DRIVE_DIR") or ".") / run_name / name
        if dry_run:
            print(f"[drive_upload] dry-run: would copy {src} -> {out_path}")
            return str(out_path)
        if not src.is_file():
            raise FileNotFoundError(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, out_path)
        print(f"[drive_upload] copied {src} -> {out_path}")
        return str(out_path)

    if backend == "apps_script":
        url, token = _require_apps_script(upload_url, token)
        manifest = _retry(lambda: _post_json(url, {"token": token, "action": "list", "run": run_name}))
        entry = next((f for f in manifest.get("files", []) if f.get("name") == name), None)
        if entry is None:
            raise FileNotFoundError(f"{name} not found in Drive run folder '{run_name}'. "
                                    f"Available: {[f.get('name') for f in manifest.get('files', [])]}")
        total = int(entry.get("parts") or 1)
        expected_size = int(entry.get("size") or 0)
        if dry_run:
            print(f"[drive_upload] dry-run: would download {name} ({expected_size} bytes, {total} part(s))")
            return str(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as dst:
            for index in range(total):
                resp = _retry(lambda: _post_json(url, {"token": token, "action": "part",
                                                        "run": run_name, "name": name,
                                                        "index": index}))
                blob = base64.b64decode(resp["data"])
                dst.write(blob)
                print(f"[drive_upload] part {index + 1}/{total} of {name}")
            dst.flush()
            os.fsync(dst.fileno())
        if expected_size and tmp.stat().st_size != expected_size:
            got = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch for {name}: got {got}, expected {expected_size}")
        os.replace(tmp, out_path)
        print(f"[drive_upload] downloaded -> {out_path}")
        return str(out_path)

    raise SystemExit(f"Unknown backend: {backend}")


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-COCOAI Drive sync (no admin required)")
    p.add_argument("--action", required=True,
                   choices=["upload", "download", "download-latest", "list"])
    p.add_argument("--backend", default=None, choices=["copy", "apps_script", "none"],
                   help="default: google drive env hint via --drive-dir, else apps_script")
    p.add_argument("--file", default=None, help="file to upload (upload action)")
    p.add_argument("--run", default="default", help="remote run folder, e.g. resnet50_binary")
    p.add_argument("--name", default="last_weights.pt", help="remote file name (download/list)")
    p.add_argument("--out", default=None, help="download destination (default: ./<name>)")
    p.add_argument("--drive-dir", default=os.environ.get("MSCOCOAI_DRIVE_DIR"))
    p.add_argument("--apps-script-url", default=os.environ.get("MSCOCOAI_APPS_SCRIPT_URL"))
    p.add_argument("--apps-script-token", default=os.environ.get("MSCOCOAI_APPS_SCRIPT_TOKEN"))
    p.add_argument("--chunk-mb", type=float, default=24.0)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    backend = args.backend or ("copy" if args.drive_dir else "apps_script")
    chunk_bytes = int(args.chunk_mb * 1024 * 1024)

    if args.action == "upload":
        if not args.file:
            raise SystemExit("--file is required for --action upload")
        upload_file(args.file, backend=backend, run_name=args.run,
                    drive_dir=args.drive_dir, upload_url=args.apps_script_url,
                    token=args.apps_script_token, chunk_bytes=chunk_bytes,
                    dry_run=args.dry_run)
    elif args.action == "list":
        manifest = list_remote(backend=backend, run_name=args.run,
                               upload_url=args.apps_script_url,
                               token=args.apps_script_token,
                               drive_dir=args.drive_dir)
        print(json.dumps(manifest, indent=2))
    elif args.action in ("download", "download-latest"):
        name = args.name if args.action == "download" else "last_weights.pt"
        out = args.out or os.path.join("outputs", args.run, name)
        download_file(backend=backend, run_name=args.run, name=name, out_path=out,
                      drive_dir=args.drive_dir, upload_url=args.apps_script_url,
                      token=args.apps_script_token, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
