"""Environment preflight: python, torch, GPU, RAM, disk, internet, dataset.

Run this first on any new machine (Kaggle session, Colab, lab PC):

    python -m mscocoai.doctor
    python -m mscocoai.doctor --out-dir outputs

Exit code 1 iff a blocker is found.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _free_disk(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def _ram_gb() -> tuple[float, float]:
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = float(v.strip().split()[0]) / (1024 * 1024)  # GB
        return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _gpu_info() -> list[dict]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"name": props.name, "vram_gb": round(props.total_memory / (1024 ** 3), 1)})
    return gpus


def _internet(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mscocoai-doctor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403, 429), f"HTTP {exc.code}"  # reachable but refused/limited
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc}"


def collect_report(out_dir: str | None = None) -> dict:
    out_dir = Path(out_dir or os.getcwd())
    report: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        report["torch"] = None
        report["cuda"] = False
        report["torch_error"] = str(exc)
    report["gpu"] = _gpu_info()
    total_ram, avail_ram = _ram_gb()
    report["ram_gb"] = {"total": round(total_ram, 1), "available": round(avail_ram, 1)}
    report["disk"] = {
        "root": round(_free_disk(Path("/")), 1),
        "out_dir": round(_free_disk(_resolve_writer_root(out_dir)), 1),
    }
    hf_ok, hf_msg = _internet("https://huggingface.co")
    report["internet"] = {"huggingface.co": {"ok": hf_ok, "detail": hf_msg}}
    drive_url = os.environ.get("MSCOCOAI_APPS_SCRIPT_URL")
    if drive_url:
        drive_ok, drive_msg = _internet(drive_url)
        report["internet"]["apps_script"] = {"ok": drive_ok, "detail": drive_msg}
    # dataset presence
    ds = os.environ.get("MSCOCOAI_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE") \
        or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    report["paths"] = {"out_dir": str(out_dir.resolve()), "hf_cache": ds, "out_dir_writable": True}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".doctor_probe"
        probe.write_text("ok")
        probe.unlink()
    except Exception as exc:  # noqa: BLE001
        report["paths"]["out_dir_writable"] = False
        report["paths"]["out_dir_error"] = str(exc)

    blockers: list[str] = []
    if not report["torch"]:
        blockers.append("torch is not installed — run `pip install -r requirements.txt`")
    if report["disk"]["out_dir"] < 5.0:
        blockers.append(f"free disk on out_dir is only {report['disk']['out_dir']:.1f} GB (need ≥ 5 GB)")
    if not report["internet"]["huggingface.co"]["ok"]:
        blockers.append("huggingface.co unreachable — check network/proxy or download offline and pass --data-dir")
    report["blockers"] = blockers
    return report


def _resolve_writer_root(path: Path) -> Path:
    """Walk up until a usable disk exists (avoids failing on restricted paths)."""
    cur = path.resolve()
    while True:
        if cur.exists():
            return cur
        parent = cur.parent
        if parent == cur:
            return Path(".").resolve()
        cur = parent


def render(report: dict) -> str:
    lines: list[str] = []
    if report.get("blockers"):
        lines.append("!!! BLOCKERS:")
        lines += [f"  - {b}" for b in report["blockers"]]
        lines.append("")
    lines.append("== python ==")
    lines.append(f"  version={report['python']}  executable={report['executable']}")
    lines.append("== torch ==")
    lines.append(f"  torch={report.get('torch')}  cuda={report.get('cuda')}")
    lines.append("== gpu ==")
    gpus = report.get("gpu") or []
    if gpus:
        for i, g in enumerate(gpus):
            lines.append(f"  gpu[{i}] name={g['name']} vram={g['vram_gb']} GB")
    else:
        lines.append("  none (CPU-only run will be very slow; use --device cpu and --limit)")
    lines.append("== ram ==")
    lines.append(f"  total={report['ram_gb']['total']} GB  available={report['ram_gb']['available']} GB")
    lines.append("== disk ==")
    lines.append(f"  / -> {report['disk']['root']} GB free   out_dir -> {report['disk']['out_dir']} GB free")
    lines.append("== internet ==")
    for name, info in report.get("internet", {}).items():
        mark = "OK" if info["ok"] else "FAIL"
        lines.append(f"  {name}: {mark} ({info['detail']})")
    lines.append("== paths ==")
    for k, v in report.get("paths", {}).items():
        lines.append(f"  {k}: {v}")
    if not report.get("blockers"):
        lines.append("== verdict == OK — no blockers found")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-COCOAI environment preflight")
    p.add_argument("--out-dir", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = collect_report(args.out_dir)
    print(render(report))
    return 1 if report.get("blockers") else 0


if __name__ == "__main__":
    sys.exit(main())
