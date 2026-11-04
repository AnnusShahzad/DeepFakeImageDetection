"""Durable, resumable run results: history.jsonl, metrics.csv, results.json, summary.md.

Rows are keyed rather than blindly appended, so re-running an interrupted epoch
replaces that epoch's row instead of duplicating it. The reader tolerates a
truncated final line from a crash-during-append.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# row schema:
#   epoch row   -> {"type": "epoch", "epoch": int, "train_loss": float, "val_loss": float,
#                   "lr": float, "is_best": bool, "elapsed_sec": float, "val_metrics": dict}
#   partial row -> {"type": "partial", "epoch": int, "step": int, "train_loss": float}


def _atomic_write(path: str | os.PathLike, content: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _key(row: dict[str, Any]) -> tuple:
    row_type = row.get("type", "row")
    if row_type == "epoch":
        return ("epoch", int(row.get("epoch", -1)))
    if row_type == "partial":
        return ("partial", int(row.get("epoch", -1)), int(row.get("step", -1)))
    return ("row", id(row))


def _trim_trailing_partial_line(path: Path) -> int:
    """If history.jsonl ends mid-line (crash during append), drop the partial
    line so future appends don't glue onto it. Returns number of bytes dropped."""
    if not path.is_file():
        return 0
    data = path.read_bytes()
    if data.endswith(b"\n") or not data:
        return 0
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        path.write_bytes(b"")
        return len(data)
    path.write_bytes(data[: last_nl + 1])
    return len(data) - last_nl - 1


def _parse_history(path: Path) -> list[dict[str, Any]]:
    """Read history.jsonl, tolerating a truncated final line."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # truncated final line from a crash during append — drop it,
                # keep every earlier (complete) row
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


class ResultsLog:
    """Owns the four artefacts written into a run directory."""

    def __init__(self, out_dir: str | os.PathLike, *, run_name: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or self.out_dir.name
        self.history_path = self.out_dir / "history.jsonl"
        _trim_trailing_partial_line(self.history_path)
        self.csv_path = self.out_dir / "metrics.csv"
        self.json_path = self.out_dir / "results.json"
        self.summary_path = self.out_dir / "summary.md"
        self._rows: list[dict[str, Any]] = _parse_history(self.history_path)
        # De-duplicate any pre-existing rows that share a key (keep the last).
        deduped: dict[tuple, dict[str, Any]] = {}
        for row in self._rows:
            deduped[_key(row)] = row
        self._rows = list(deduped.values())

    # ------------------------------------------------------------------ rows
    def rows(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows]

    def epoch_rows(self) -> list[dict[str, Any]]:
        rows = [dict(r) for r in self._rows if r.get("type") == "epoch"]
        return sorted(rows, key=lambda r: r.get("epoch", -1))

    def partial_rows(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows if r.get("type") == "partial"]

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        key = _key(row)
        replaced = False
        for i, existing in enumerate(self._rows):
            if _key(existing) == key:
                self._rows[i] = row
                replaced = True
                break
        if not replaced:
            self._rows.append(row)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row

    def last_epoch(self) -> int:
        epochs = [int(r.get("epoch", -1)) for r in self._rows if r.get("type") == "epoch"]
        return max(epochs) if epochs else -1

    # ------------------------------------------------------------- artifacts
    def flush(self, cfg: dict[str, Any] | None = None) -> None:
        epochs = self.epoch_rows()
        self._write_csv(epochs)
        self._write_json(epochs, cfg or {})
        self._write_summary(epochs, cfg or {})

    def _write_csv(self, epochs: list[dict[str, Any]]) -> None:
        header = ["epoch", "train_loss", "val_loss", "lr", "is_best",
                  "elapsed_sec", "accuracy", "macro_f1", "auc", "f1"]
        lines = [",".join(header)]
        for row in epochs:
            vm = row.get("val_metrics") or {}
            parts = [
                str(row.get("epoch", "")),
                f"{row.get('train_loss', '')}" if row.get("train_loss") is not None else "",
                f"{row.get('val_loss', '')}" if row.get("val_loss") is not None else "",
                f"{row.get('lr', '')}" if row.get("lr") is not None else "",
                str(bool(row.get("is_best", False))),
                f"{row.get('elapsed_sec', '')}" if row.get("elapsed_sec") is not None else "",
                f"{vm.get('accuracy', '')}",
                f"{vm.get('macro_f1', '')}",
                f"{vm.get('auc', '')}",
                f"{vm.get('f1', '')}",
            ]
            lines.append(",".join(parts))
        _atomic_write(self.csv_path, "\n".join(lines) + "\n")

    def _best(self, epochs: list[dict[str, Any]]) -> tuple[int | None, float | None]:
        best_epoch, best_val = None, None
        for row in epochs:
            val = row.get("val_loss")
            if val is None:
                continue
            if best_val is None or float(val) < best_val:
                best_val = float(val)
                best_epoch = int(row.get("epoch", 0))
        return best_epoch, best_val

    def _write_json(self, epochs: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
        best_epoch, best_val = self._best(epochs)
        payload = {
            "run_name": self.run_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": cfg,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "epochs": epochs,
            "stop_reason": cfg.get("stop_reason"),
            "threshold": cfg.get("threshold"),
        }
        _atomic_write(self.json_path, json.dumps(payload, indent=2, default=str))

    def _write_summary(self, epochs: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
        best_epoch, best_val = self._best(epochs)
        lines = [
            f"# Run summary — {self.run_name}",
            "",
            f"- task: {cfg.get('task', '?')}  backbone: {cfg.get('backbone', '?')}",
            f"- stop_reason: {cfg.get('stop_reason', 'completed')}",
            f"- best_epoch: {best_epoch}  best_val: {best_val}",
            "",
            "| epoch | train_loss | val_loss | lr | accuracy | macro_f1 |",
            "|---|---|---|---|---|---|",
        ]
        for row in epochs:
            vm = row.get("val_metrics") or {}
            lines.append(
                "| {e} | {tr} | {vl} | {lr} | {acc} | {mf} |".format(
                    e=row.get("epoch", ""),
                    tr=f"{row.get('train_loss', ''):.4f}" if isinstance(row.get("train_loss"), (int, float)) else "",
                    vl=f"{row.get('val_loss', ''):.4f}" if isinstance(row.get("val_loss"), (int, float)) else "",
                    lr=f"{row.get('lr', ''):.2e}" if isinstance(row.get("lr"), (int, float)) else "",
                    acc=f"{vm['accuracy']:.4f}" if isinstance(vm.get("accuracy"), (int, float)) else "",
                    mf=f"{vm['macro_f1']:.4f}" if isinstance(vm.get("macro_f1"), (int, float)) else "",
                )
            )
        _atomic_write(self.summary_path, "\n".join(lines) + "\n")
