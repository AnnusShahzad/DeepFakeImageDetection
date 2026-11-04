"""Parallel job queue for a shared PC: several runs without collisions.

Each job runs with its own output directory, log file and slot in
jobs_status.json. CPU workers are budgeted per job; CUDA devices are pinned
per job when ``gpu`` is set; jobs launch in their own session so they survive
the SSH session; the launcher is restartable — it re-reads the status file and
skips jobs that already completed.

    python -m mscocoai.launcher --jobs jobs.json --max-concurrency 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import shutil


@dataclass
class Job:
    name: str
    cmd: list[str]
    out_dir: str
    gpu: int | None = None
    workers: int = 4


def estimate_concurrency(free_vram_gb: float, per_job_vram_gb: float = 10.0) -> int:
    if free_vram_gb <= 0:
        return 1
    return max(1, int(free_vram_gb // per_job_vram_gb))


def find_free_vram_gb() -> float:
    """Free VRAM in GB, or 0 when nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return 0.0
    vals = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line:
            try:
                vals.append(float(line.split()[0]) / 1024.0)
            except ValueError:
                pass
    return min(vals) if vals else 0.0


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_status(status_path: Path) -> dict:
    if not status_path.is_file():
        return {}
    try:
        return json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_status(status_path: Path, status: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(status_path, json.dumps(status, indent=2))


def run_jobs(jobs: list[Job], *, max_concurrency: int = 1, log_dir: str | os.PathLike = "logs",
             status_path: str | os.PathLike = "jobs_status.json", dry_run: bool = False) -> int:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_path)
    status = _load_status(status_path)

    remaining = [j for j in jobs if status.get(j.name, {}).get("status") != "done"]
    if not remaining:
        print(f"[launcher] all {len(jobs)} job(s) already done; nothing to run")
        return len(jobs)
    if dry_run:
        for j in remaining:
            print(f"[launcher] DRY RUN {j.name}: {' '.join(j.cmd)}")
        return len(remaining)

    def run_one(job: Job) -> int:
        log_path = log_dir / f"{job.name}.log"
        started = time.time()
        status[job.name] = {"status": "running", "started_at": started,
                            "out_dir": job.out_dir, "log": str(log_path)}
        _save_status(status_path, status)
        env = os.environ.copy()
        if job.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        env["OMP_NUM_THREADS"] = str(job.workers)
        env["MKL_NUM_THREADS"] = str(job.workers)
        with log_path.open("w") as fh:
            fh.write(f"# job={job.name} cmd={' '.join(job.cmd)}\n")
            fh.flush()
            try:
                result = subprocess.run(job.cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
                rc = result.returncode
            except Exception as exc:  # noqa: BLE001
                rc = 127
                fh.write(f"\n[launcher] failed to start: {exc}\n")
        status[job.name] = {
            "status": "done" if rc == 0 else "failed",
            "returncode": rc,
            "started_at": started,
            "finished_at": time.time(),
            "out_dir": job.out_dir,
            "log": str(log_path),
        }
        _save_status(status_path, status)
        print(f"[launcher] {job.name}: {'OK' if rc == 0 else f'FAILED (rc={rc})'} -> {log_path}")
        return rc

    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
        return_codes = list(pool.map(run_one, remaining))
    succeeded = sum(1 for rc in return_codes if rc == 0)
    print(f"[launcher] finished {succeeded}/{len(remaining)} remaining job(s)")
    return succeeded


def load_jobs(path: str | os.PathLike) -> list[Job]:
    data = json.loads(Path(path).read_text())
    jobs: list[Job] = []
    for i, entry in enumerate(data):
        try:
            jobs.append(Job(name=str(entry["name"]), cmd=[str(c) for c in entry["cmd"]],
                            out_dir=str(entry["out_dir"]), gpu=entry.get("gpu"),
                            workers=int(entry.get("workers", 4))))
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"jobs[{i}]: invalid entry ({exc}); expected "
                             f"{{name, cmd, out_dir, gpu?, workers?}}") from exc
    return jobs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run several MS-COCOAI jobs in parallel")
    p.add_argument("--jobs", required=True, help="JSON file: [{name, cmd, out_dir, gpu?, workers?}]")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="default: estimated from free GPU VRAM (10 GB per job)")
    p.add_argument("--per-job-vram-gb", type=float, default=10.0)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--status-path", default="jobs_status.json")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    jobs = load_jobs(args.jobs)
    concurrency = args.max_concurrency
    if concurrency is None:
        concurrency = estimate_concurrency(find_free_vram_gb(), args.per_job_vram_gb)
        print(f"[launcher] estimated max concurrency: {concurrency} (free VRAM)")
    run_jobs(jobs, max_concurrency=concurrency, log_dir=args.log_dir,
             status_path=args.status_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
