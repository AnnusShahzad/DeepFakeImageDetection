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


def collect(run_dirs, reports_root: Path | None = None) -> list[dict[str, Any]]:
    test_reports = collect_test_reports(reports_root) if reports_root else {}
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
        name = payload.get("run_name", path.name)
        report = test_reports.get(name) or {}
        t_det = report.get("binary") or {}
        t_att = report.get("attribution") or {}
        row = {
            "run": name,
            "task": cfg.get("task"),
            "backbone": cfg.get("backbone"),
            "freq_branch": cfg.get("freq_branch", "none"),
            "clip_init": cfg.get("clip_init", "none"),
            "holdout_generator": cfg.get("holdout_generator"),
            "resize_mode": cfg.get("resize_mode", "squash"),
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
            "test_accuracy": t_det.get("accuracy"),
            "test_balanced_accuracy": t_det.get("balanced_accuracy"),
            "test_f1": t_det.get("f1"),
            "test_auc": t_det.get("auc"),
            "test_attr_accuracy": t_att.get("accuracy"),
            "test_macro_f1": t_att.get("macro_f1"),
            "lr": next(
                (e.get("lr") for e in (payload.get("epochs") or []) if e.get("epoch") == payload.get("best_epoch")),
                None,
            ),
            "path": str(path),
            "per_generator": report.get("per_generator_detection") or {},
        }
        rows.append(row)
    return rows


def collect_test_reports(reports_root: Path | None) -> dict[str, dict[str, Any]]:
    """Map run name -> report.json produced by mscocoai.evaluate."""
    if not reports_root or not reports_root.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for report_path in sorted(reports_root.glob("*/report.json")):
        payload = _read_results(report_path)
        if not payload:
            continue
        run = report_path.parent.name
        for suffix in ("_test", "_jpeg75", "_blur"):
            if run.endswith(suffix):
                run = run[: -len(suffix)]
                break
        out.setdefault(run, payload)
    return out


_COLUMNS = ["run", "task", "backbone", "freq_branch", "clip_init", "holdout_generator",
            "resize_mode", "epochs_done", "best_epoch", "best_val", "stop_reason", "threshold",
            "accuracy", "balanced_accuracy", "macro_f1", "auc", "attr_accuracy",
            "test_accuracy", "test_balanced_accuracy", "test_f1", "test_auc",
            "test_attr_accuracy", "test_macro_f1", "path"]


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
    lines = ["| Run | Task | Backbone | freq | clip | holdout | resize | best_ep | val bal_acc | val AUC | val macroF1 | test acc | test bal_acc | test AUC | test attr acc | test macroF1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        def fmt(key: str) -> str:
            val = row.get(key)
            return f"{val:.4f}" if isinstance(val, (int, float)) else "—"

        cells = [
            str(row.get("run", "")),
            str(row.get("task", "")),
            str(row.get("backbone", "")),
            str(row.get("freq_branch", "")),
            str(row.get("clip_init", "")),
            str(row.get("holdout_generator", "") or "—"),
            str(row.get("resize_mode", "squash")),
            str(row.get("best_epoch", "")),
            fmt("balanced_accuracy"),
            fmt("auc"),
            fmt("macro_f1"),
            fmt("test_accuracy"),
            fmt("test_balanced_accuracy"),
            fmt("test_auc"),
            fmt("test_attr_accuracy"),
            fmt("test_macro_f1"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # ---- O5: per-generator detection difficulty on the test split ----
    per_gen_rows = [(r["run"], g, m) for r in rows for g, m in (r.get("per_generator") or {}).items()]
    if per_gen_rows:
        generators = sorted({g for _, g, _ in per_gen_rows})
        lines += ["", "## Per-generator detection on the test split (O5)", "",
                  "| Run | " + " | ".join(generators) + " |",
                  "|---" * (len(generators) + 1) + "|"]
        by_run: dict[str, dict[str, Any]] = {}
        for run, gen, metrics in per_gen_rows:
            by_run.setdefault(run, {})[gen] = metrics
        for run, metrics_by_gen in by_run.items():
            cells = []
            for gen in generators:
                m = metrics_by_gen.get(gen) or {}
                bal = m.get("balanced_accuracy")
                cells.append(f"{bal:.4f}" if isinstance(bal, (int, float)) else "—")
            lines.append(f"| {run} | " + " | ".join(cells) + " |")
        lines += ["", "Cells are balanced accuracy for *real vs one generator* "
                  "(the per-generator binary task).", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate MS-COCOAI run results")
    p.add_argument("run_dirs", nargs="+", help="directories containing results.json")
    p.add_argument("--csv", default="results/ablation.csv")
    p.add_argument("--md", default="results/ablation.md")
    p.add_argument("--reports", default=None,
                   help="report root written by mscocoai.evaluate (adds test metrics)")
    args = p.parse_args()
    rows = collect(args.run_dirs, Path(args.reports) if args.reports else None)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"[collect] {len(rows)} run(s) -> {args.csv}, {args.md}")


if __name__ == "__main__":
    main()
