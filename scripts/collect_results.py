"""Aggregate results.json files from many run directories into CSV + markdown.

    python scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_results(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _best_epoch_metrics(run: dict[str, Any]) -> dict[str, Any]:
    best_epoch = run.get("best_epoch")
    epochs = run.get("epochs") or []
    best = None
    for row in epochs:
        vm = row.get("val_metrics") or {}
        if best_epoch is None or row.get("epoch") == best_epoch:
            best = vm
    return best or {}


def collect(run_dirs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in run_dirs:
        path = Path(entry)
        if not path.is_dir():
            continue
        payload = _read_results(path / "results.json")
        if not payload:
            print(f"[collect] skipping {path}: no results.json")
            continue
        cfg = payload.get("config") or {}
        vm = _best_epoch_metrics(payload)
        det = vm.get("detection") or vm
        att = vm.get("attribution") or {}
        row = {
            "run": payload.get("run_name", path.name),
            "task": cfg.get("task"),
            "backbone": cfg.get("backbone"),
            "freq_branch": cfg.get("freq_branch", "none"),
            "clip_init": cfg.get("clip_init", "none"),
            "holdout_generator": cfg.get("holdout_generator"),
            "epochs_done": len(payload.get("epochs") or []),
            "best_epoch": payload.get("best_epoch"),
            "best_val": payload.get("best_val"),
            "stop_reason": payload.get("stop_reason", "completed"),
            "threshold": payload.get("threshold"),
            "accuracy": det.get("accuracy"),
            "balanced_accuracy": det.get("balanced_accuracy"),
            "macro_f1": att.get("macro_f1", det.get("macro_f1")),
            "auc": det.get("auc"),
            "attr_accuracy": att.get("accuracy"),
            "lr": next(
                (e.get("lr") for e in (payload.get("epochs") or []) if e.get("epoch") == payload.get("best_epoch")),
                None,
            ),
            "path": str(path),
        }
        rows.append(row)
    return rows


_COLUMNS = ["run", "task", "backbone", "freq_branch", "clip_init", "holdout_generator",
            "epochs_done", "best_epoch", "best_val", "stop_reason", "threshold",
            "accuracy", "balanced_accuracy", "macro_f1", "auc", "attr_accuracy", "path"]


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Run | Task | Backbone | freq | clip | holdout | best_epoch | accuracy | balanced_acc | macro_f1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        cells = [
            str(row.get("run", "")),
            str(row.get("task", "")),
            str(row.get("backbone", "")),
            str(row.get("freq_branch", "")),
            str(row.get("clip_init", "")),
            str(row.get("holdout_generator", "") or "—"),
            str(row.get("best_epoch", "")),
            f"{row['accuracy']:.4f}" if isinstance(row.get("accuracy"), (int, float)) else "—",
            f"{row['balanced_accuracy']:.4f}" if isinstance(row.get("balanced_accuracy"), (int, float)) else "—",
            f"{row['macro_f1']:.4f}" if isinstance(row.get("macro_f1"), (int, float)) else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate MS-COCOAI run results")
    p.add_argument("run_dirs", nargs="+", help="directories containing results.json")
    p.add_argument("--csv", default="results/ablation.csv")
    p.add_argument("--md", default="results/ablation.md")
    args = p.parse_args()
    rows = collect(args.run_dirs)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"[collect] {len(rows)} run(s) -> {args.csv}, {args.md}")


if __name__ == "__main__":
    main()
