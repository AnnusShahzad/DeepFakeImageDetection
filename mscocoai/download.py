"""Parallel, resumable download of the MS-COCOAI dataset.

Uses the HF cache, which keeps *.incomplete partials, so an interrupted
download resumes instead of restarting. Chunk-parallel transfer is enabled
via hf_transfer when it is installed (set before huggingface_hub is imported).

Optionally exports the parquet shards to a folder mirror (verbatim image
bytes, no re-encoding) so the training dataloader can read plain JPEGs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mscocoai.constants import HF_DATASET_ID, LABEL_B_NAMES

if importlib.util.find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_snapshot(repo_id: str, cache_dir: str | os.PathLike | None = None,
                      workers: int = 8, revision: str | None = None,
                      allow_patterns: list[str] | None = None) -> Path:
    """Download the dataset snapshot; returns the local cache path."""
    from huggingface_hub import snapshot_download

    patterns = allow_patterns or ["data/*.parquet", "README.md"]
    local = snapshot_download(repo_id=repo_id, repo_type="dataset",
                              cache_dir=str(cache_dir) if cache_dir else None,
                              allow_patterns=patterns, max_workers=int(workers),
                              revision=revision)
    return Path(local)


def _split_from_shard(name: str) -> str:
    stem = name.split(".", 1)[0]
    for split in ("train", "validation", "test"):
        if stem.lower().startswith(split):
            return split
    return "train"


def _sniff_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] == b"GIF87a" or data[:6] == b"GIF89a":
        return ".gif"
    return ".bin"


def _extract_bytes(raw) -> bytes:
    if isinstance(raw, dict):
        data = raw.get("bytes")
        if data is None:
            path = raw.get("path")
            if isinstance(path, str) and Path(path).is_file():
                data = Path(path).read_bytes()
        if data is None:
            raise ValueError("parquet Image cell contains neither bytes nor readable path")
        return data
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    raise ValueError(f"unsupported Image cell type: {type(raw)}")


def _export_one(shard: Path, dest: Path, resume: bool = True) -> dict:
    import pyarrow.parquet as pq

    split = _split_from_shard(shard.name)
    table = pq.read_table(shard, columns=["Image", "Label_B"])
    images = table.column("Image").to_pylist()
    labels = table.column("Label_B").to_pylist()
    written = skipped = bytes_written = unknown = 0
    for row, (raw, label_b) in enumerate(zip(images, labels)):
        try:
            data = _extract_bytes(raw)
        except ValueError:
            unknown += 1
            continue
        try:
            cls = LABEL_B_NAMES[int(label_b)]
        except (ValueError, IndexError, TypeError):
            unknown += 1
            continue
        out_dir = dest / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = _sniff_ext(data)
        target = out_dir / f"{shard.stem}_{row:05d}{ext}"
        if resume and target.is_file() and target.stat().st_size == len(data):
            skipped += 1
            continue
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        written += 1
        bytes_written += len(data)
    return {"shard": str(shard), "split": split, "written": written,
            "skipped": skipped, "unknown": unknown, "bytes": bytes_written}


def export_images(parquet_paths, dest, *, workers: int = 8, resume: bool = True) -> dict:
    """Export parquet shards to <dest>/<split>/<class_name>/<shard>_<row>.<ext>."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shards = list(parquet_paths)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(_export_one, Path(s), dest, resume): s for s in shards}
        for fut, shard in futures.items():
            try:
                results.append(fut.result())
                print(f"[export] {Path(shard).name}: done")
            except Exception as exc:  # noqa: BLE001 — report and continue with the other shards
                print(f"[export] {Path(shard).name}: FAILED ({exc})")
                results.append({"shard": str(shard), "split": "?", "written": 0,
                                "skipped": 0, "unknown": 0, "bytes": 0, "error": str(exc)})
    summary = {
        "shards": len(results),
        "written": sum(r.get("written", 0) for r in results),
        "skipped": sum(r.get("skipped", 0) for r in results),
        "unknown": sum(r.get("unknown", 0) for r in results),
        "bytes": sum(r.get("bytes", 0) for r in results),
        "dest": str(dest),
        "results": results,
    }
    (dest / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def verify_sizes(local_root: str | os.PathLike, repo_id: str = HF_DATASET_ID,
                 revision: str | None = None) -> list[str]:
    """Return the list of repo files whose size differs from the local copy."""
    from huggingface_hub import HfApi

    root = Path(local_root)
    api = HfApi()
    mismatches: list[str] = []
    for info in  api.list_repo_tree(repo_id, repo_type="dataset", revision=revision,
                                    recursive=True):
        if not info.size:
            continue
        local = root / info.path
        if not local.is_file():
            mismatches.append(f"missing: {info.path}")
        elif local.stat().st_size != info.size:
            mismatches.append(f"size mismatch: {info.path} (local {local.stat().st_size} vs repo {info.size})")
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download the MS-COCOAI dataset (parallel, resumable)")
    p.add_argument("--repo-id", default=HF_DATASET_ID)
    p.add_argument("--revision", default=None)
    p.add_argument("--cache-dir", default=None,
                   help="HF hub cache directory (default: HF_HOME / ~/.cache/huggingface)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (use 8-16 on fast links)")
    p.add_argument("--allow-patterns", default=None,
                   help="comma-separated glob patterns (default: data/*.parquet, README.md)")
    p.add_argument("--export-images", default=None, metavar="DIR",
                   help="export parquet shards to a folder mirror (verbatim bytes)")
    p.add_argument("--export-workers", type=int, default=4,
                   help="threads used by --export-images")
    p.add_argument("--verify", action="store_true",
                   help="verify downloaded file sizes against the repo metadata")
    return p


def main() -> None:
    args = build_parser().parse_args()
    patterns = [p.strip() for p in args.allow_patterns.split(",")] if args.allow_patterns else None
    print(f"[download] repo={args.repo_id} workers={args.workers} cache_dir={args.cache_dir or '<default>'}")
    local = download_snapshot(args.repo_id, cache_dir=args.cache_dir, workers=args.workers,
                              revision=args.revision, allow_patterns=patterns)
    print(f"[download] complete -> {local}")

    shards = sorted((local / "data").glob("*.parquet")) if (local / "data").is_dir() else []
    if args.export_images:
        if not shards:
            raise SystemExit(f"no parquet shards found under {local / 'data'}")
        print(f"[export] exporting {len(shards)} shard(s) to {args.export_images}")
        summary = export_images(shards, args.export_images, workers=args.export_workers)
        print(f"[export] written={summary['written']} skipped={summary['skipped']} "
              f"unknown={summary['unknown']} bytes={summary['bytes']}")

    if args.verify:
        mismatches = verify_sizes(local, args.repo_id, args.revision)
        if mismatches:
            print("[verify] mismatches:")
            for m in mismatches:
                print(" ", m)
            raise SystemExit(1)
        print("[verify] all files match the repo metadata")


if __name__ == "__main__":
    main()
