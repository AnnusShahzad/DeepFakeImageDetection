# MS-COCOAI SOTA Profile & Reliability Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing MS-COCOAI pipeline into a crash-proof, multi-device, multi-platform system that can be handed to team members and that has a real chance of beating the published 0.80 / 0.45 MS-COCOAI baselines.

**Architecture:** One importable package `mscocoai/` with module CLIs, extended in three phases. Phase A makes work durable (keyed result log, resumable download, three data sources, preflight doctor). Phase B adds the composable SOTA profile (frequency branch, CLIP branch, TTA, calibration, balanced metrics, LOGO protocol) and multi-device execution (DDP, TPU). Phase C makes it handable (parallel launcher, result collector, quickstart, per-platform guides, notebook). The existing paper-faithful path is never touched: every new behaviour is behind an explicit flag, and `scripts/run_paper_models.sh` keeps reproducing the paper exactly.

**Tech Stack:** Python 3.10+, PyTorch 2.x, timm, open_clip_torch, huggingface_hub (snapshot_download + hf_transfer), datasets/pyarrow, numpy, scikit-learn, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-mscocoai-sota-profile-design.md`

**Why one plan, not three:** the three phases are not independent subsystems — `engine.py` and `train.py` are touched by results persistence, the time budget, DDP, and every new model component. Splitting would mean three sequential rewrites of the same two files. The plan is therefore one document with three phase groups and strict phase ordering.

## Global Constraints

- **Do not break the existing 36 tests.** `tests/test_engine.py` constructs `TrainConfig` with explicit keyword arguments → **every new config field must be appended last with a default value**. `tests/test_engine.py::test_one_epoch_writes_artifacts` asserts `history.jsonl` has **exactly one line per epoch** → no meta/start/stop rows may be written to `history.jsonl`; run metadata lives in `results.json`.
- **No Google Cloud project, service account, OAuth client, or API key.** Drive access stays `copy` (Drive for Desktop / mounted Drive) or `apps_script`.
- **No secrets in git, in logs, or in `print()` output.** Secrets come from env vars / Kaggle Secrets only.
- **Paper-faithful defaults unchanged.** `scripts/run_paper_models.sh` must still reproduce O1–O5 exactly; every new component defaults to off.
- **Atomic writes everywhere:** temp file → `fsync` → `os.replace`.
- **Test command:** `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q` (venv has torch 2.15 CPU, timm, open_clip, datasets, sklearn, pytest).
- **Python:** system python is PEP-668 externally managed — always use the venv interpreter above.
- **Reporting protocol:** the headline comparison against 0.80 / 0.45 uses the **balanced** protocol; raw-skew accuracy is always reported beside it, never alone.
- **Commit only after the user approves.** Nothing is committed until then; `../docs` modifications are the user's pre-existing edits and must not be touched.

## Review Focus

Five input classes the spec implies but no task's tests would otherwise exercise. Each gets a test, in the task listed:

1. **A truncated final line in `history.jsonl`** (crash mid-append) → the reader must drop only that line and keep every earlier row. → Task 1
2. **Two runs of the same epoch** (crash after checkpoint, before log write) → the log must contain one row for that epoch, not two, and no gap. → Task 1
3. **A checkpoint written before this change** (no `model_spec`, no `step_in_epoch`) → must still load and rebuild the legacy model. → Task 11
4. **A dataset directory containing parquet shards but no images** → must be detected and read, not silently treated as an empty folder mirror. → Task 5
5. **A run with `--holdout-generator` where validation also contains that generator** → the generator must be excluded from validation too, or early stopping leaks the unseen generator. → Task 14

---

## File Structure

| File | Responsibility |
|---|---|
| `mscocoai/results.py` | **new** — keyed result log: `history.jsonl`, `metrics.csv`, `results.json`, `summary.md` |
| `mscocoai/download.py` | **new** — parallel/resumable HF snapshot download, image export, verify |
| `mscocoai/doctor.py` | **new** — environment preflight (GPU, RAM, disk, internet, dataset, secrets) |
| `mscocoai/distributed.py` | **new** — DDP init/helpers, cross-rank metric gathering |
| `mscocoai/xla.py` | **new** — TPU device abstraction over `torch_xla` |
| `mscocoai/frequency.py` | **new** — DCT-II basis, de-normalisation, high-pass, `FrequencyBranch` |
| `mscocoai/tta.py` | **new** — deterministic test-time augmentation views + probability averaging |
| `mscocoai/calibrate.py` | **new** — validation-fitted decision threshold |
| `mscocoai/launcher.py` | **new** — parallel job queue for a shared PC |
| `mscocoai/models.py` | modify — `MultiTaskNet`, `ClipMultiTask`, `FrequencyBranch` wiring, `model_spec`, `build_model_from_spec` |
| `mscocoai/engine.py` | modify — result log, resume reconciliation, time budget, DDP, new components, threshold |
| `mscocoai/train.py` | modify — new flags; `TrainConfig` gains trailing defaulted fields |
| `mscocoai/data.py` | modify — `ParquetImageDataset`, `--data-layout`, LOGO filtering, worker budget |
| `mscocoai/metrics.py` | modify — balanced accuracy, `balanced_binary_report` |
| `mscocoai/evaluate.py` | modify — TTA, stored threshold, balanced report, model_spec rebuild |
| `mscocoai/ensemble.py` | modify — mixed architectures, per-head weights, TTA |
| `scripts/collect_results.py` | **new** — aggregate `results.json` from many runs into CSV + markdown |
| `scripts/quickstart.sh` | **new** — platform-detecting one-command entry point |
| `scripts/run_ablation.sh` | **new** — the ablation grid (scaled + full) |
| `scripts/train_ddp.sh`, `scripts/train_tpu.sh` | **new** — launchers |
| `requirements-lock.txt` | **new** — pinned versions |
| `colab/mscocoai_colab.ipynb`, `colab/COLAB_GUIDE.md` | **new** — Colab target |
| `TEAM_GUIDE.md` | **new** — handover guide |
| `README.md`, `kaggle/KAGGLE_GUIDE.md`, `lab_pc/A4000_SETUP.md` | modify — updated for every new command |
| `tests/test_results.py`, `test_download.py`, `test_distributed.py`, `test_frequency.py`, `test_tta.py`, `test_calibrate.py`, `test_launcher.py`, `test_doctor.py` | **new** |

---

# Phase A — Durability, data, preflight

### Task 1: Keyed result log

**Files:**
- Create: `mscocoai/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  ```python
  class ResultsLog:
      def __init__(self, out_dir: str | Path, *, run_name: str | None = None) -> None
      def append(self, row: dict) -> dict
      def rows(self) -> list[dict]
      def epoch_rows(self) -> list[dict]
      def flush(self, cfg: dict | None = None) -> None
      def last_epoch(self) -> int
      @property
      def history_path(self) -> Path
  ```
  Row schema — epoch row: `{"type": "epoch", "epoch": int, "train_loss": float, "val_loss": float, "lr": float, "is_best": bool, "elapsed_sec": float, "val_metrics": dict, "ts": str}`; partial row: `{"type": "partial", "epoch": int, "step": int, "train_loss": float, "ts": str}`.
  Files written in `out_dir`: `history.jsonl`, `metrics.csv`, `results.json`, `summary.md`.

- [ ] **Step 1: Write the failing test** — `tests/test_results.py`:
  ```python
  def test_epoch_row_is_upserted_not_duplicated(tmp_path):
      log = ResultsLog(tmp_path)
      log.append({"type": "epoch", "epoch": 0, "train_loss": 1.0, "val_loss": 0.5,
                  "lr": 1e-4, "is_best": True, "elapsed_sec": 1.0, "val_metrics": {}})
      log.append({"type": "epoch", "epoch": 0, "train_loss": 0.9, "val_loss": 0.4,
                  "lr": 1e-4, "is_best": True, "elapsed_sec": 1.0, "val_metrics": {}})
      assert len(log.epoch_rows()) == 1
      assert log.epoch_rows()[0]["val_loss"] == 0.4

  def test_truncated_final_line_is_dropped_and_earlier_rows_kept(tmp_path):
      (tmp_path / "history.jsonl").write_text(
          '{"type": "epoch", "epoch": 0, "val_loss": 0.5}\n'
          '{"type": "epoch", "epoch": 1, "val_loss": 0.4')
      log = ResultsLog(tmp_path)
      assert [r["epoch"] for r in log.epoch_rows()] == [0]

  def test_partial_rows_are_keyed_by_epoch_and_step(tmp_path):
      log = ResultsLog(tmp_path)
      log.append({"type": "partial", "epoch": 0, "step": 10, "train_loss": 0.7})
      log.append({"type": "partial", "epoch": 0, "step": 10, "train_loss": 0.6})
      assert len(log.rows()) == 1
      assert log.rows()[0]["train_loss"] == 0.6

  def test_flush_writes_all_four_artifacts(tmp_path):
      log = ResultsLog(tmp_path)
      log.append({"type": "epoch", "epoch": 0, "train_loss": 1.0, "val_loss": 0.5,
                  "lr": 1e-4, "is_best": True, "elapsed_sec": 1.0,
                  "val_metrics": {"accuracy": 0.9, "macro_f1": 0.8}})
      log.flush({"task": "binary"})
      for name in ("history.jsonl", "metrics.csv", "results.json", "summary.md"):
          assert (tmp_path / name).is_file()
      import json
      res = json.loads((tmp_path / "results.json").read_text())
      assert res["best_val"] == 0.5 and res["best_epoch"] == 0
      assert "accuracy" in (tmp_path / "metrics.csv").read_text()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_results.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'mscocoai.results'`

- [ ] **Step 3: Implement `mscocoai/results.py`**
  - `__init__` stores `out_dir` (created if missing), `run_name`, and loads existing rows via a tolerant reader: split on `"\n"`, try `json.loads` per line, **skip** unparseable lines but keep every earlier valid one.
  - `append(row)`: compute key — `("epoch", row["epoch"])` for `type == "epoch"`, `("partial", row["epoch"], row["step"])` for `type == "partial"`, else `("row", len(rows))`; replace an existing row with the same key, else append. Set `row["ts"]` to an ISO-8601 UTC timestamp if absent. Append to `history.jsonl` with `open("a")`, `write(json.dumps(row) + "\n")`, `flush()`, `os.fsync()`. Rewrite the in-memory row list.
  - `epoch_rows()`: rows with `type == "epoch"`, sorted by `epoch`.
  - `flush(cfg=None)`: rewrite `metrics.csv` (header `epoch,train_loss,val_loss,lr,is_best,elapsed_sec,accuracy,macro_f1,auc,f1` then one line per epoch row, flattened from `val_metrics` via `.get`), `results.json` (`run_name`, `updated_at`, `config`, `best_epoch`, `best_val`, `epochs`, `stop_reason` — all via tmp+`os.replace`), and `summary.md` (a markdown table of the epoch rows plus a best-epoch line).
  - `last_epoch()`: max epoch among epoch rows, or `-1`.
  - Atomic helper shared with `models.py`: write to `path.with_suffix(path.suffix + ".tmp")`, `os.fsync`, `os.replace`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_results.py -v`
  Expected: 4 passed

- [ ] **Step 5: Run the full suite to confirm nothing broke**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 40 passed (36 existing + 4 new)

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/results.py tests/test_results.py
  git commit -m "feat(results): keyed, crash-tolerant result log"
  ```

### Task 2: Wire the result log into the training loop + crash-safe resume

**Files:**
- Modify: `mscocoai/engine.py` (the loop at lines 244–334, `_save_all` at 119–140, resume block at 209–239)
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `ResultsLog` (Task 1).
- Produces:
  - `engine._save_all(model, optimizer, scheduler, scaler, ema, cfg, epoch, best_val, out_dir, is_best, val_metrics, step_in_epoch=0, sampler_epoch=0)` — also stores `"step_in_epoch"`, `"sampler_epoch"`, `"model_spec"`, `"threshold"` in the checkpoint state.
  - `engine.fit(cfg) -> dict` now returns `{"best_val", "best_epoch", "out_dir", "stop_reason", "history_path"}`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_engine.py`:
  ```python
  def test_history_has_one_row_per_epoch_after_resume(tiny_dataset, tmp_path):
      out = tmp_path / "run_rows"
      fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
      fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
      rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
      assert [r["epoch"] for r in rows] == [0, 1]
      assert all(r["type"] == "epoch" for r in rows)

  def test_results_json_and_summary_exist(tiny_dataset, tmp_path):
      out = tmp_path / "run_json"
      fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
      res = json.loads((out / "results.json").read_text())
      assert res["best_epoch"] == 0 and res["best_val"] is not None
      assert (out / "summary.md").is_file() and (out / "metrics.csv").is_file()

  def test_resume_after_truncated_history_line(tiny_dataset, tmp_path):
      out = tmp_path / "run_trunc"
      fit(_cfg(tiny_dataset, out, epochs=1, resume="none"))
      with (out / "history.jsonl").open("a") as fh:
          fh.write('{"type": "epoch", "epoch": 1, "val_loss": 0.1')
      fit(_cfg(tiny_dataset, out, epochs=2, resume="auto"))
      rows = [json.loads(l) for l in (out / "history.jsonl").read_text().strip().splitlines()]
      assert [r["epoch"] for r in rows] == [0, 1]
  ```
  Note: `tests/test_engine.py` already imports `json` and `Path`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_engine.py -v`
  Expected: FAIL — `results.json` missing, rows lack `"type"`.

- [ ] **Step 3: Implement the engine changes**
  - In `fit()`, create `log = ResultsLog(out_dir, run_name=Path(cfg.out_dir).name)`.
  - Replace the manual `history_path.open("a")` write with `log.append(row)` where `row` gains `"type": "epoch"`.
  - After each epoch's checkpoint save, call `log.flush(cfg=config_dict)` where `config_dict = cfg.as_dict() if hasattr(cfg, "as_dict") else vars(cfg)`.
  - `_save_all(...)` gains `step_in_epoch: int = 0` and `sampler_epoch: int = 0` parameters; the state dict gains `"step_in_epoch": step_in_epoch`, `"sampler_epoch": sampler_epoch`, `"model_spec": getattr(cfg, "model_spec", None)`, `"threshold": getattr(cfg, "threshold", None)`.
  - Mid-epoch `--save-steps` checkpoint call passes `step_in_epoch=micro`, `sampler_epoch=epoch`; the end-of-epoch call passes `step_in_epoch=0`, `sampler_epoch=epoch`.
  - Resume block: after loading the checkpoint, compute `resume_step = int(ckpt.get("step_in_epoch", 0) or 0)`; `start_epoch = int(ckpt.get("epoch", -1)) + 1` when `resume_step == 0`, else `start_epoch = int(ckpt.get("epoch", -1))`. Store `resume_step` for the loop. Print `start_epoch` and `resume_step`.
  - In the training loop, when `epoch == start_epoch` and `resume_step > 0`, skip the first `resume_step` batches of `train_loader` (iterate and `continue`), then reset `resume_step = 0`.
  - `fit()` returns `{"best_val": best_val, "best_epoch": best_epoch, "out_dir": str(out_dir), "stop_reason": stop_reason, "history_path": str(log.history_path)}`; `stop_reason` is `"completed"` unless stopped earlier.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_engine.py -v`
  Expected: all pass, including the 3 pre-existing tests.

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 43 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/engine.py tests/test_engine.py
  git commit -m "feat(engine): durable result log and crash-safe resume"
  ```

### Task 3: Session time budget + Drive upload of results

**Files:**
- Modify: `mscocoai/engine.py` (`_maybe_upload`, the batch loop), `mscocoai/train.py`
- Modify: `mscocoai/drive_upload.py` (add `upload_files`)
- Test: `tests/test_engine.py`, `tests/test_drive_upload.py`

**Interfaces:**
- Consumes: `ResultsLog` (Task 1).
- Produces:
  - `drive_upload.upload_files(paths: list[Path], *, backend: str, run_name: str, drive_dir: str | None = None, upload_url: str | None = None, token: str | None = None) -> list[str]` — uploads each file, skipping ones already uploaded with the same size, never raising (returns the list of successfully uploaded paths).
  - `TrainConfig` gains trailing defaults: `max_hours: float = 0.0`, `upload_metrics: bool = True`.
  - `engine.fit` honours `cfg.max_hours` and `cfg.upload_metrics`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_engine.py`:
  ```python
  def test_max_hours_stops_training_and_records_reason(tiny_dataset, tmp_path):
      out = tmp_path / "run_budget"
      cfg = _cfg(tiny_dataset, out, epochs=10, resume="none")
      cfg.max_hours = 1e-9   # budget already exhausted
      result = fit(cfg)
      assert result["stop_reason"] == "time_budget"
      assert (out / "last.pt").is_file()

  def test_upload_metrics_uploads_history_files(tiny_dataset, tmp_path, monkeypatch):
      out = tmp_path / "run_upload"
      cfg = _cfg(tiny_dataset, out, epochs=1, resume="none")
      cfg.drive_backend = "copy"
      cfg.drive_dir = str(tmp_path / "drive")
      cfg.upload_every = "epoch"
      cfg.upload_metrics = True
      seen = []
      monkeypatch.setattr("mscocoai.drive_upload.upload_files",
                          lambda paths, **kw: seen.extend(str(p) for p in paths) or [])
      fit(cfg)
      assert any(p.endswith("history.jsonl") for p in seen)
      assert any(p.endswith("results.json") for p in seen)
  ```
  And to `tests/test_drive_upload.py`:
  ```python
  def test_upload_files_copies_each_file(tmp_path):
      src = tmp_path / "a.txt"; src.write_text("x")
      dest = tmp_path / "drive"
      out = drive_upload.upload_files([src], backend="copy", run_name="r", drive_dir=str(dest))
      assert out and (dest / "r" / "a.txt").read_text() == "x"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_engine.py tests/test_drive_upload.py -v`
  Expected: FAIL — `max_hours` not a field, `upload_files` missing.

- [ ] **Step 3: Implement**
  - `train.py`: add `--max-hours` (float, default 0.0) and `--upload-metrics` (flag, default on via `argparse.BooleanOptionalAction`, so `--no-upload-metrics` disables it). Append `max_hours: float = 0.0` and `upload_metrics: bool = True` to `TrainConfig` **after** the existing fields.
  - `engine.fit`: record `t_start = time.time()`; inside the batch loop, after each optimizer step, check `if cfg.max_hours and (time.time() - t_start) > cfg.max_hours * 3600:` → set `stop_reason = "time_budget"`, `break` out of the batch loop.
  - After the loop, when `stop_reason == "time_budget"`: save a checkpoint with `step_in_epoch=micro`, append a partial row `{"type": "partial", "epoch": epoch, "step": micro, "train_loss": running / max(1, seen)}`, `log.flush(...)`, `_maybe_upload(...)`, and `break` the epoch loop.
  - `_maybe_upload(out_dir, cfg)`: build the file list — always `last_weights.pt`; when `cfg.upload_metrics` also `history.jsonl`, `metrics.csv`, `results.json` (only files that exist). Call `drive_upload.upload_files(files, backend=cfg.drive_backend, run_name=..., drive_dir=..., upload_url=..., token=...)` inside a `try/except` that prints a warning and continues (never crash training because of Drive).
  - `drive_upload.upload_files`: loop over paths, call the existing `upload_file(...)` per path inside a `try/except`, collect successes, return them. Skip a file whose destination already exists with the same size for the `copy` backend.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_engine.py tests/test_drive_upload.py -v`
  Expected: all pass

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 46 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/engine.py mscocoai/train.py mscocoai/drive_upload.py tests/
  git commit -m "feat(engine): time budget and Drive upload of result logs"
  ```

### Task 4: Parallel, resumable dataset downloader

**Files:**
- Create: `mscocoai/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `mscocoai.constants.HF_DATASET_ID`.
- Produces:
  ```python
  def export_images(parquet_paths: list[Path], dest: Path, *, workers: int = 8,
                    resume: bool = True) -> dict
  def verify_files(paths: list[Path], expected: dict[str, int]) -> list[str]
  def build_parser() -> argparse.ArgumentParser
  def main() -> None
  ```
  CLI: `python -m mscocoai.download [--workers N] [--cache-dir DIR] [--export-images DIR] [--verify] [--repo-id ID] [--limit-shards N]`.
  `export_images` returns `{"written": int, "skipped": int, "bytes": int}` and writes `dest/manifest.json`.

- [ ] **Step 1: Write the failing test** — `tests/test_download.py`:
  ```python
  def test_export_images_writes_class_folders_and_is_resumable(tmp_path):
      import pyarrow as pa, pyarrow.parquet as pq
      from PIL import Image
      import io
      src = tmp_path / "data"; src.mkdir()
      img = Image.new("RGB", (8, 8), (1, 2, 3))
      buf = io.BytesIO(); img.save(buf, format="JPEG")
      rows = {"Image": [buf.getvalue()], "Label_A": [1], "Label_B": [2]}
      pq.write_table(pa.table(rows), src / "train-00000-of-00001.parquet")
      dest = tmp_path / "mirror"
      out = download.export_images([src / "train-00000-of-00001.parquet"], dest, workers=1)
      assert out["written"] == 1
      assert list((dest / "train" / "sdxl").glob("*.jpg"))
      again = download.export_images([src / "train-00000-of-00001.parquet"], dest, workers=1)
      assert again["skipped"] == 1 and again["written"] == 0

  def test_export_resumes_after_interruption(tmp_path):
      # same parquet, but a manifest claiming one file already written
      ...  # build parquet, call export twice, assert the second call writes 0
  ```
  Use `pyarrow` (already a transitive dep of `datasets`); if it is missing, `pytest.skip`.

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_download.py -v`
  Expected: FAIL — `ModuleNotFoundError: No module named 'mscocoai.download'`

- [ ] **Step 3: Implement `mscocoai/download.py`**
  - At import time, **before** importing `huggingface_hub`, set `os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")` if `importlib.util.find_spec("hf_transfer")` is not None. Do the `huggingface_hub` import inside the function, not at module scope.
  - `download_snapshot(repo_id, cache_dir=None, workers=8) -> Path`: call `snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir, allow_patterns=["data/*.parquet", "README.md"], max_workers=workers)`. The HF cache keeps `*.incomplete` files, so an interrupted call resumes.
  - `export_images(parquet_paths, dest, workers=8, resume=True)`: for each shard, read it with `pyarrow.parquet.read_table`; for each row, decode the image bytes (they are the original encoded bytes — write them verbatim, sniffing the extension from the magic bytes: `\xff\xd8` → `.jpg`, `\x89PNG` → `.png`, `RIFF` → `.webp`, else `.bin`); target path `dest/<split>/<class_name>/<shardstem>_<rowindex>.<ext>` where `<split>` comes from the shard filename prefix (`train-`/`validation-`/`test-`) and `<class_name>` is `LABEL_B_NAMES[label_b]`. Use a `concurrent.futures.ProcessPoolExecutor(max_workers=workers)` over shards; write to `path.with_suffix(".tmp")` then `os.replace`. If `resume` and the target already exists, skip it and count it in `skipped`. Write `dest/manifest.json` with `{"shards": [...], "written": n, "bytes": n}` after each shard completes.
  - `verify_files(paths, expected)`: return the list of paths whose size differs from `expected`.
  - `main()`: parse args, call `download_snapshot`, print the cache path, then if `--export-images` call `export_images` over the downloaded `data/*.parquet` files, then if `--verify` compare sizes against the HF tree API (`huggingface_hub.HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)`) and print mismatches. Print a final summary of bytes and MiB/s.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_download.py -v`
  Expected: 2 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 48 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/download.py tests/test_download.py
  git commit -m "feat(download): parallel resumable dataset download and image export"
  ```

### Task 5: Third data source — parquet directory

**Files:**
- Modify: `mscocoai/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from Task 4 (independent code path; the parquet layout is read directly).
- Produces:
  ```python
  class ParquetImageDataset(Dataset):
      def __init__(self, root: str | os.PathLike, split: str, transform=None,
                   limit: int | None = None, holdout_generator: str | None = None)
      def __len__(self) -> int
      def __getitem__(self, index: int) -> dict
      def labels_b(self) -> np.ndarray
      def raw_image(self, index: int)
  def detect_layout(root: Path) -> str          # "parquet" | "folder" | "unknown"
  def build_dataset(split, *, data_dir, limit, norm, img_size, degradation="none",
                    layout="auto", holdout_generator=None) -> Dataset
  ```

- [ ] **Step 1: Write the failing tests** — append to `tests/test_data.py`:
  ```python
  def test_detect_layout_distinguishes_parquet_and_folder(tmp_path):
      assert data_mod.detect_layout(tmp_path) == "unknown"
      (tmp_path / "data").mkdir(); (tmp_path / "data" / "train-00000-of-00001.parquet").write_bytes(b"x")
      assert data_mod.detect_layout(tmp_path) == "parquet"
      (tmp_path / "train").mkdir()
      assert data_mod.detect_layout(tmp_path) == "folder"

  def test_parquet_dataset_reads_images_and_labels(tmp_path):
      # write a tiny parquet with one Image/Label_A/Label_B row
      ...
      ds = data_mod.ParquetImageDataset(tmp_path, "train", transform=None)
      assert len(ds) == 1
      item = ds[0]
      assert item["label_b"] == 2 and item["label_a"] == 1
      assert item["image"].size == (8, 8)

  def test_holdout_generator_excludes_that_class(tmp_path):
      ds = data_mod.ParquetImageDataset(tmp_path, "train", transform=None, holdout_generator="sdxl")
      assert all(lb != 2 for lb in ds.labels_b())
  ```
  Add `import mscocoai.data as data_mod` at the top of the test file.

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_data.py -v`
  Expected: FAIL — `detect_layout` / `ParquetImageDataset` missing

- [ ] **Step 3: Implement**
  - `detect_layout(root)`: `"parquet"` if `root/"data"` contains `*.parquet` or `root` contains `*.parquet`; `"folder"` if any of `root/train`, `root/validation`, `root/test` is a directory; else `"unknown"`.
  - `ParquetImageDataset`: at init, glob `root/"data"/f"{split}-*.parquet"` (fall back to `root/f"{split}-*.parquet"`), sort them, read each with `pyarrow.parquet.read_table`, and build an index of `(shard_path, row_index, label_a, label_b)` — read the label columns eagerly (cheap) and decode image bytes lazily in `__getitem__`. Apply `holdout_generator` filtering by label-b name. Apply `limit` after filtering.
  - `__getitem__` decodes bytes with `PIL.Image.open(io.BytesIO(...)).convert("RGB")`, applies `self.transform` if set, and returns `{"image", "label_a", "label_b", "index"}` — identical to `LocalFolderDataset`.
  - `raw_image(index)` returns the un-transformed PIL image.
  - `build_dataset`: when `data_dir` is set, resolve `layout` — `"auto"` uses `detect_layout(Path(data_dir))`; `"parquet"` forces `ParquetImageDataset`; `"folder"` forces `LocalFolderDataset`; `"unknown"` raises `FileNotFoundError` with a message naming both layouts. Pass `holdout_generator` through. Add the `layout` and `holdout_generator` keyword parameters with defaults so the existing call sites keep working.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_data.py -v`
  Expected: all pass (existing 8 + 3 new)

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 51 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/data.py tests/test_data.py
  git commit -m "feat(data): parquet-directory data source with generator holdout"
  ```

### Task 6: Environment preflight (`doctor`) + device/precision/disk flags

**Files:**
- Create: `mscocoai/doctor.py`
- Modify: `mscocoai/train.py`, `mscocoai/engine.py`, `mscocoai/data.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: Task 5's `build_dataset(..., layout=...)`.
- Produces:
  ```python
  # mscocoai/doctor.py
  def collect_report(out_dir: str | None = None) -> dict
  def render(report: dict) -> str
  def main() -> int          # returns 0 when no blocker, 1 when a blocker is present
  # mscocoai/data.py
  def resolve_num_workers(requested: int | str, device: str) -> int
  # mscocoai/engine.py
  def resolve_device(spec: str) -> str
  def check_free_disk(path: str | os.PathLike, min_free_gb: float) -> tuple[bool, float]
  ```
  `TrainConfig` gains trailing defaults: `device: str = "auto"`, `precision: str = "auto"`, `min_free_gb: float = 5.0`, `channels_last: bool = False`.

- [ ] **Step 1: Write the failing tests** — `tests/test_doctor.py`:
  ```python
  def test_resolve_num_workers_auto_scales_with_cpu():
      assert resolve_num_workers("auto", "cpu") >= 0
      assert resolve_num_workers(3, "cpu") == 3
      assert resolve_num_workers("auto", "cuda") >= 2

  def test_check_free_disk_flags_insufficient_space(tmp_path):
      ok, free = check_free_disk(tmp_path, min_free_gb=1e9)
      assert ok is False and free >= 0

  def test_collect_report_has_expected_sections():
      rep = doctor.collect_report()
      for key in ("python", "torch", "cuda", "gpu", "ram_gb", "disk", "internet", "paths"):
          assert key in rep
      assert isinstance(doctor.render(rep), str)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_doctor.py -v`
  Expected: FAIL — modules missing

- [ ] **Step 3: Implement**
  - `data.resolve_num_workers(requested, device)`: `int` → return as-is (≥0); `"auto"` → `min(8, max(2, (os.cpu_count() or 2) // 2))`, and `0` when `os.cpu_count() <= 2`.
  - `engine.resolve_device(spec)`: `"auto"` → `"cuda" if torch.cuda.is_available() else "cpu"`; `"cuda"`/`"cuda:N"` → validated; `"cpu"` → `"cpu"`; `"tpu"` → `"tpu"` (handled by Task 8).
  - `engine.check_free_disk(path, min_free_gb)` → `(ok, free_gb)` using `shutil.disk_usage`.
  - `engine.fit`: call `check_free_disk(out_dir, cfg.min_free_gb)`; if not ok, print a clear error and `raise SystemExit(...)`. Use `resolve_device(cfg.device)` and `resolve_num_workers(cfg.num_workers, device)` for the loaders. Keep `num_workers: int = 4` as the dataclass field but allow `int | str` — change the annotation to `int | str = 4` (existing test passes `num_workers=0`, still valid).
  - `train.py`: add `--device`, `--precision`, `--min-free-gb`, `--channels-last`; append the four fields to `TrainConfig` with defaults.
  - `doctor.collect_report(out_dir=None)`: gather python version, torch version, `torch.cuda.is_available()`, GPU name(s) via `torch.cuda.get_device_name(0)`, total VRAM in GB, RAM total/available in GB (`/proc/meminfo` or `psutil` if present, else `os.sysconf`), free disk on `/` and on `out_dir` (or cwd), internet reachability of `https://huggingface.co` and of the Apps Script URL if `MSCOCOAI_APPS_SCRIPT_URL` is set (2 s timeout, never raise), dataset presence (does `HF_DATASETS_CACHE`/`HF_HOME` hold the dataset, or does `--data-dir` exist), and writability of the output dir. Return a dict with a `"blockers"` list of human-readable strings.
  - `doctor.render(report)`: a plain-text report, one section per key, blockers listed first.
  - `doctor.main()`: print the rendered report; `return 1 if report["blockers"] else 0`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_doctor.py -v`
  Expected: 3 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 54 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/doctor.py mscocoai/train.py mscocoai/engine.py mscocoai/data.py tests/test_doctor.py
  git commit -m "feat(env): preflight doctor and device/disk/precision controls"
  ```

---

# Phase B — SOTA profile and multi-device execution

### Task 7: Distributed data parallel

**Files:**
- Create: `mscocoai/distributed.py`
- Modify: `mscocoai/engine.py`, `mscocoai/data.py`, `mscocoai/train.py`
- Test: `tests/test_distributed.py`

**Interfaces:**
- Consumes: Task 2's `fit()` structure, Task 6's `resolve_device`.
- Produces:
  ```python
  # mscocoai/distributed.py
  def is_distributed() -> bool
  def get_rank() -> int
  def get_world_size() -> int
  def is_main_process() -> bool
  def init_distributed(backend: str | None = None) -> tuple[int, int, int]
  def barrier() -> None
  def all_gather_np(arr: np.ndarray) -> np.ndarray
  def cleanup_distributed() -> None
  def build_sampler(dataset, *, shuffle: bool, seed: int, drop_last: bool = False)
  ```
  `init_distributed` returns `(rank, world_size, local_rank)`; when `RANK` is absent from the environment it returns `(0, 1, 0)` and initialises nothing.

- [ ] **Step 1: Write the failing tests** — `tests/test_distributed.py`:
  ```python
  def test_not_distributed_by_default():
      assert not distributed.is_distributed()
      assert distributed.get_rank() == 0 and distributed.get_world_size() == 1
      assert distributed.is_main_process()

  def test_init_distributed_is_noop_without_env():
      assert distributed.init_distributed() == (0, 1, 0)

  def test_all_gather_np_single_process():
      import numpy as np
      out = distributed.all_gather_np(np.array([1.0, 2.0]))
      assert list(out) == [1.0, 2.0]

  def test_build_sampler_plain_dataset_without_distributed(tiny_dataset):
      ds = build_dataset("train", data_dir=str(tiny_dataset), limit=None, norm="imagenet", img_size=64)
      sampler = distributed.build_sampler(ds, shuffle=True, seed=1)
      assert sampler is not None
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_distributed.py -v`
  Expected: FAIL — module missing

- [ ] **Step 3: Implement**
  - `mscocoai/distributed.py`: read `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from `os.environ`; `init_distributed` picks `nccl` when CUDA is available else `gloo`, calls `torch.distributed.init_process_group(backend=backend)` when `WORLD_SIZE > 1`, and sets `torch.cuda.set_device(local_rank)` on CUDA. `barrier()` is a no-op when not distributed. `all_gather_np` uses `dist.all_gather` on CPU tensors and concatenates along axis 0. `build_sampler` returns `DistributedSampler(dataset, num_replicas, rank, shuffle, seed, drop_last)` when distributed, else `None`.
  - `data.build_loader`: accept an optional `sampler` parameter; when given, use it instead of `shuffle`/`balanced_sampler` (a `WeightedRandomSampler` and a `DistributedSampler` are mutually exclusive — raise `ValueError` if both are requested).
  - `engine.fit`: call `init_distributed()` at the top; build the train sampler with `build_sampler(train_ds, shuffle=True, seed=cfg.seed, drop_last=True)` and the val sampler with `build_sampler(val_ds, shuffle=False, seed=cfg.seed)`; call `sampler.set_epoch(epoch)` each epoch; wrap the model in `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)` when `world_size > 1`; guard every write (checkpoint save, result-log append, `log.flush`, Drive upload) with `is_main_process()`; compute the validation loss as the world-size-weighted mean of each rank's shard loss via `dist.all_reduce`; gather validation scores/labels with `all_gather_np` before computing metrics so the report covers the whole split.
  - `train.py`: only rank 0 writes `config.json`; add `--local-rank` is **not** needed (torchrun sets `LOCAL_RANK`).

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_distributed.py -v`
  Expected: 4 passed

- [ ] **Step 5: Verify a real 2-process gloo run**
  Run:
  ```bash
  /tmp/opencode/venv-mscocoai/bin/torchrun --nproc_per_node=2 --master_port=29517 \
    -m mscocoai.train --task binary --backbone resnet18 --data-dir <tiny_dataset> \
    --img-size 64 --epochs 1 --batch-size 4 --num-workers 0 --out-dir /tmp/opencode/ddp_out
  ```
  Expected: exits 0, and `/tmp/opencode/ddp_out/history.jsonl` contains **exactly one** epoch row (not two).

- [ ] **Step 6: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 58 passed

- [ ] **Step 7: Commit**
  ```bash
  git add mscocoai/distributed.py mscocoai/engine.py mscocoai/data.py mscocoai/train.py tests/test_distributed.py
  git commit -m "feat(ddp): multi-GPU training with rank-0-only artifact writes"
  ```

### Task 8: TPU device path (torch_xla)

**Files:**
- Create: `mscocoai/xla.py`
- Modify: `mscocoai/engine.py`, `mscocoai/train.py`
- Create: `scripts/train_tpu.sh`
- Test: `tests/test_xla.py`

**Interfaces:**
- Consumes: Task 6's `resolve_device`, Task 7's `init_distributed`.
- Produces:
  ```python
  # mscocoai/xla.py
  def xla_available() -> bool
  def xla_device() -> str
  def xla_seed(seed: int) -> None
  def xla_mark_step() -> None
  def xla_amp_context() -> contextlib.AbstractContextManager
  ```

- [ ] **Step 1: Write the failing test** — `tests/test_xla.py`:
  ```python
  def test_xla_helpers_are_importable_without_tpu():
      import mscocoai.xla as xla
      assert xla.xla_available() in (True, False)
      assert isinstance(xla.xla_device(), str)
      xla.xla_seed(42)      # must not raise even without torch_xla
      xla.xla_mark_step()   # must not raise
      with xla.xla_amp_context():
          pass
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_xla.py -v`
  Expected: FAIL — module missing

- [ ] **Step 3: Implement**
  - `mscocoai/xla.py`: import `torch_xla` lazily inside each function; when it is unavailable, every helper becomes a no-op and `xla_available()` returns `False`. `xla_amp_context()` returns `torch.autocast("xla", dtype=torch.bfloat16)` when `torch_xla` is importable, else `contextlib.nullcontext()`.
  - `engine`: when the resolved device is `"tpu"`, use `xla.xla_device()` as the torch device, call `xla.xla_seed(cfg.seed)`, wrap the forward pass in `xla.xla_amp_context()`, disable `GradScaler` (XLA uses bf16 autocast), call `xla.xla_mark_step()` after each optimizer step, and use `gloo` as the DDP backend (XLA uses `xla` backend only for multi-device; single-device TPU v3-8 is one process).
  - `train.py`: accept `--device tpu`.
  - `scripts/train_tpu.sh`: a documented launcher that checks for `torch_xla`, sets `PJRT_DEVICE=TPU`, and runs `python -m mscocoai.train --device tpu ...`.
  - **Document clearly** in the module docstring and in the guides that this path is **untested on real TPU hardware** — no TPU was available in the development environment.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_xla.py -v`
  Expected: 1 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 59 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/xla.py mscocoai/engine.py mscocoai/train.py scripts/train_tpu.sh tests/test_xla.py
  git commit -m "feat(tpu): torch_xla device path (untested on TPU hardware)"
  ```

### Task 9: Frequency-aware dual-branch model

**Files:**
- Create: `mscocoai/frequency.py`
- Modify: `mscocoai/models.py`, `mscocoai/train.py`
- Test: `tests/test_frequency.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except `mscocoai.constants.IMAGENET_MEAN/IMAGENET_STD`.
- Produces:
  ```python
  # mscocoai/frequency.py
  def dct_matrix(n: int) -> torch.Tensor            # (n, n) orthonormal DCT-II basis
  def dct2d(x: torch.Tensor, k: int) -> torch.Tensor # (B, k, k) low-frequency coefficients of grayscale
  def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor
  def gaussian_blur(x: torch.Tensor, sigma: float = 1.0, radius: int = 2) -> torch.Tensor
  def highpass(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor
  class FrequencyBranch(nn.Module):
      def __init__(self, mode: str = "dct", k: int = 32, out_dim: int = 256,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD)
      def forward(self, x: torch.Tensor) -> torch.Tensor   # (B, out_dim)
  # mscocoai/models.py
  class MultiTaskNet(nn.Module):                        # generalised MultiTaskEffB0
      def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dropout=0.2,
                   freq_branch="none", freq_k=32, freq_dim=256,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD)
      def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]
  def build_model(task, backbone, pretrained=True, *, freq_branch="none", freq_k=32,
                  clip_init="none", img_size=224, mean=IMAGENET_MEAN,
                  std=IMAGENET_STD) -> nn.Module
  ```
  `MultiTaskEffB0` must remain importable and behave exactly as before (alias of `MultiTaskNet` with the old defaults).

- [ ] **Step 1: Write the failing tests** — `tests/test_frequency.py`:
  ```python
  def test_dct_matrix_is_orthonormal():
      import numpy as np
      from scipy.fft import dct
      m = frequency.dct_matrix(8).numpy()
      assert np.allclose(m @ m.T, np.eye(8), atol=1e-5)

  def test_dct2d_matches_scipy_on_a_small_image():
      import numpy as np
      from scipy.fft import dct
      img = np.arange(16, dtype=np.float64).reshape(4, 4)
      ref = dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")
      got = frequency.dct2d(torch.from_numpy(img)[None, None, :, :], k=4)[0, 0].numpy()
      assert np.allclose(got, ref[:4, :4], atol=1e-4)

  def test_denormalize_roundtrips():
      x = torch.rand(2, 3, 8, 8)
      mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
      normed = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
      assert torch.allclose(frequency.denormalize(normed, mean, std), x, atol=1e-5)

  def test_frequency_branch_output_shape():
      branch = frequency.FrequencyBranch(mode="both", k=16, out_dim=64)
      out = branch(torch.randn(2, 3, 64, 64))
      assert out.shape == (2, 64)

  def test_multitask_with_freq_branch_shapes():
      model = models.build_model("multitask", "resnet18", pretrained=False, freq_branch="dct", freq_k=16)
      zd, za = model(torch.randn(2, 3, 64, 64))
      assert zd.shape == (2,) and za.shape == (2, 6)
  ```
  `scipy` is available (scikit-learn depends on it); skip the scipy tests if it is missing.

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_frequency.py -v`
  Expected: FAIL — module missing

- [ ] **Step 3: Implement `mscocoai/frequency.py`**
  - `dct_matrix(n)`: build the orthonormal DCT-II basis `m[i, j] = cos(pi * (j + 0.5) * i / n)`, scaled by `sqrt(1/n)` for `i == 0` and `sqrt(2/n)` otherwise. Cache per `n` in a module-level dict.
  - `dct2d(x, k)`: `x` is `(B, 1, H, W)` in **grayscale 0..1**. Resize to a power-of-two-ish size? No — use the natural size. Apply the DCT along both axes: `coeffs = m_h @ x @ m_w.T` using `torch.einsum`, then take `coeffs[..., :k, :k]`. Return `(B, k, k)`. Guard `k <= min(H, W)` by clamping `k = min(k, H, W)`.
  - `denormalize(x, mean, std)`: `x * std + mean` with per-channel broadcast.
  - `gaussian_blur(x, sigma, radius)`: separable convolution with a 1-D Gaussian kernel, applied per channel with `F.conv2d` and `replicate` padding.
  - `highpass(x, sigma)`: `x - gaussian_blur(x, sigma)`.
  - `FrequencyBranch.__init__(mode, k, out_dim, mean, std)`: store `mode` (`"dct"`, `"highpass"`, `"both"`), `k`, the mean/std tensors, and a small CNN per view: `Conv2d(1, 32, 3, stride=2, padding=1) → BatchNorm2d → GELU → Conv2d(32, 64, 3, stride=2, padding=1) → BatchNorm2d → GELU → AdaptiveAvgPool2d(1) → Flatten → Linear(64, out_dim//n_views)`. Concatenate the views, then `Linear(total, out_dim)`.
  - `FrequencyBranch.forward(x)`: `raw = denormalize(x, mean, std).clamp(0, 1)`; `gray = raw.mean(dim=1, keepdim=True)`; if `"dct"` in mode → `d = dct2d(gray, k)` → `log1p(abs(d))` → normalise per sample → CNN; if `"highpass"` in mode → `h = highpass(gray)` → CNN; concatenate and project.

- [ ] **Step 4: Implement `mscocoai/models.py` changes**
  - Rename the class to `MultiTaskNet` with the new parameters; keep `MultiTaskEffB0 = MultiTaskNet`-compatible by defining
    ```python
    class MultiTaskEffB0(MultiTaskNet):
        def __init__(self, backbone="efficientnet_b0", pretrained=True, hidden_dropout=0.2):
            super().__init__(backbone=backbone, pretrained=pretrained, hidden_dropout=hidden_dropout)
    ```
  - `MultiTaskNet.forward`: `h = self.features(x)`; when a frequency branch exists, `h = torch.cat([h, self.freq(x)], dim=1)`; then `dropout` → the two heads. The heads' input dim grows by `freq_dim`.
  - `build_model(task, backbone, pretrained=True, *, freq_branch="none", freq_k=32, clip_init="none", img_size=224, mean=..., std=...)`: for `task == "multitask"` build `MultiTaskNet(backbone, pretrained, freq_branch=freq_branch, freq_k=freq_k, mean=mean, std=std)`; `clip_init != "none"` is handled in Task 10.

- [ ] **Step 5: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_frequency.py -v`
  Expected: 5 passed

- [ ] **Step 6: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 64 passed

- [ ] **Step 7: Commit**
  ```bash
  git add mscocoai/frequency.py mscocoai/models.py mscocoai/train.py tests/test_frequency.py
  git commit -m "feat(model): frequency-aware dual-branch multi-task model"
  ```

### Task 10: CLIP foundation-model branch

**Files:**
- Modify: `mscocoai/models.py`, `mscocoai/train.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Task 9's `build_model` signature (adds `clip_init`).
- Produces:
  ```python
  class ClipMultiTask(nn.Module):
      def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                   hidden_dropout: float = 0.2, img_size: int = 224,
                   unfreeze: str = "last2")
      def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]
  def build_model(..., clip_init: str = "none", ...)   # clip_init in {"none","clip_vitb16","clip_vitl14"}
  ```
  `clip_init` maps to `("ViT-B-16", "openai")` and `("ViT-L-14", "openai")`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_models.py`:
  ```python
  def test_clip_multitask_builds_and_forward_shapes(monkeypatch):
      import mscocoai.models as m
      fake = _FakeClip()          # a tiny stand-in exposing encode_image + num_features
      monkeypatch.setattr(m, "_create_open_clip", lambda name, pretrained: fake)
      model = m.build_model("multitask", "resnet18", pretrained=False, clip_init="clip_vitb16")
      zd, za = model(torch.randn(2, 3, 64, 64))
      assert zd.shape == (2,) and za.shape == (2, 6)
  ```
  `_FakeClip` is a `nn.Module` with `encode_image(x) -> Tensor(2, 32)` and a `visual` attribute; it stands in for the real CLIP so the test needs no download.

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_models.py -v`
  Expected: FAIL — `_create_open_clip` / `clip_init` missing

- [ ] **Step 3: Implement**
  - `models._create_open_clip(model_name, pretrained)` — a module-level helper that calls `open_clip.create_model_and_transforms(model_name, pretrained=pretrained)` and returns the model. It exists so tests can monkeypatch it.
  - `models.ClipMultiTask`: builds the encoder via `_create_open_clip`; extracts the feature dim by running `encode_image` on a zero tensor; `unfreeze` ∈ `{"frozen", "last2", "full"}` controls `requires_grad` (for `"last2"`, unfreeze the last two transformer blocks, `ln_post` and `proj` — mirror `clip_probe.py`'s logic but resolve block indices from the model rather than hard-coding `resblocks.10/11`); heads: `Linear(feat, 1)` and `Linear(feat, 6)`; `forward` returns `(sigmoid-ready logit vector of shape (B,), (B, 6))`.
  - `build_model`: when `clip_init != "none"`, return `ClipMultiTask(model_name, pretrained, unfreeze="last2")` for `task == "multitask"`; for `task == "binary"`/`"attribution"`, wrap the CLIP encoder in a single-head module with the same output contract.
  - `train.py`: add `--clip-init` (`none|clip_vitb16|clip_vitl14`), `--clip-unfreeze` (`frozen|last2|full`), `--grad-checkpointing`; append the fields to `TrainConfig` with defaults.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_models.py -v`
  Expected: all pass

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 65 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/models.py mscocoai/train.py tests/test_models.py
  git commit -m "feat(model): CLIP foundation-model branch for multi-task training"
  ```

### Task 11: Self-describing checkpoints (`model_spec`)

**Files:**
- Modify: `mscocoai/models.py`, `mscocoai/engine.py`, `mscocoai/evaluate.py`, `mscocoai/ensemble.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Task 9/10's `build_model` kwargs.
- Produces:
  ```python
  # mscocoai/models.py
  def model_spec(task: str, backbone: str, *, freq_branch="none", freq_k=32,
                 clip_init="none", img_size=224, norm="imagenet",
                 mean=IMAGENET_MEAN, std=IMAGENET_STD) -> dict
  def build_model_from_spec(spec: dict, pretrained: bool = False) -> nn.Module
  def load_model_for_eval(checkpoint, device="cpu") -> tuple[nn.Module, dict]
  ```
  `load_model_for_eval` rebuilds from `ckpt["model_spec"]` when present, else falls back to the legacy `task`/`backbone` fields.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_models.py`:
  ```python
  def test_model_spec_roundtrip_rebuilds_same_architecture():
      spec = models.model_spec("multitask", "resnet18", freq_branch="dct", freq_k=16)
      m1 = models.build_model_from_spec(spec)
      m2 = models.build_model_from_spec(spec)
      assert set(m1.state_dict()) == set(m2.state_dict())

  def test_load_model_for_eval_uses_model_spec(tmp_path):
      model = models.build_model("multitask", "resnet18", pretrained=False, freq_branch="dct", freq_k=16)
      spec = models.model_spec("multitask", "resnet18", freq_branch="dct", freq_k=16)
      path = tmp_path / "ck.pt"
      models.save_checkpoint({"model": model.state_dict(), "config": {}, "task": "multitask",
                              "backbone": "resnet18", "model_spec": spec}, path)
      loaded, ckpt = models.load_model_for_eval(path)
      assert set(loaded.state_dict()) == set(model.state_dict())

  def test_load_model_for_eval_falls_back_to_legacy_checkpoint(tmp_path):
      model = models.build_backbone("resnet18", num_classes=1, pretrained=False)
      path = tmp_path / "legacy.pt"
      models.save_checkpoint({"model": model.state_dict(), "config": {}, "task": "binary",
                              "backbone": "resnet18"}, path)
      loaded, ckpt = models.load_model_for_eval(path)
      assert loaded(torch.randn(2, 3, 64, 64)).shape == (2, 1)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_models.py -v`
  Expected: FAIL — `model_spec` missing

- [ ] **Step 3: Implement**
  - `model_spec(...)` returns the dict above (all values JSON-serialisable; mean/std as lists).
  - `build_model_from_spec(spec, pretrained=False)` calls `build_model` with the spec's values.
  - `load_model_for_eval`: `spec = ckpt.get("model_spec")`; if present use `build_model_from_spec(spec)`, else the existing `build_model(task, backbone, pretrained=False)` path. Then `load_state_dict(ckpt["model"])`.
  - `engine._save_all` writes `"model_spec": model_spec(...)` built from the config.
  - `evaluate.py` and `ensemble.py` call `load_model_for_eval` (they already do) — update `ensemble.py` to rebuild each member from its own spec.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_models.py -v`
  Expected: all pass

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 68 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/models.py mscocoai/engine.py mscocoai/evaluate.py mscocoai/ensemble.py tests/test_models.py
  git commit -m "feat(models): self-describing checkpoints rebuild the exact architecture"
  ```

### Task 12: Test-time augmentation and validation calibration

**Files:**
- Create: `mscocoai/tta.py`, `mscocoai/calibrate.py`
- Modify: `mscocoai/evaluate.py`, `mscocoai/ensemble.py`, `mscocoai/engine.py`, `mscocoai/train.py`
- Test: `tests/test_tta.py`, `tests/test_calibrate.py`

**Interfaces:**
- Consumes: Task 11's `load_model_for_eval`.
- Produces:
  ```python
  # mscocoai/tta.py
  def tta_views(x: torch.Tensor, mode: str, img_size: int = 224) -> list[torch.Tensor]
  def average_binary(probs: list[torch.Tensor]) -> torch.Tensor
  def average_multiclass(probs: list[torch.Tensor]) -> torch.Tensor
  # mscocoai/calibrate.py
  def fit_threshold(y_true: np.ndarray, scores: np.ndarray,
                    objective: str = "macro_f1", grid: int = 1001) -> float
  def best_threshold_from_curve(y_true, scores, objective) -> float
  ```
  `tta_views` modes: `"none"` → `[x]`; `"flip"` → `[x, flip(x)]`; `"flip_scale"` → `[x, flip(x), scale(x, 0.875), flip(scale(x, 0.875))]` where `scale(x, s)` resizes to `round(img_size / s)` and centre-crops back to `img_size`.

- [ ] **Step 1: Write the failing tests**:
  ```python
  # tests/test_tta.py
  def test_tta_modes_produce_expected_view_counts():
      x = torch.randn(2, 3, 224, 224)
      assert len(tta.tta_views(x, "none")) == 1
      assert len(tta.tta_views(x, "flip")) == 2
      assert len(tta.tta_views(x, "flip_scale")) == 4

  def test_tta_views_preserve_shape_and_are_deterministic():
      x = torch.randn(1, 3, 224, 224)
      a = tta.tta_views(x, "flip_scale")
      b = tta.tta_views(x, "flip_scale")
      assert all(v.shape == x.shape for v in a)
      assert torch.equal(a[1], b[1])

  def test_average_multiclass_normalises():
      probs = [torch.tensor([[0.2, 0.8]]), torch.tensor([[0.6, 0.4]])]
      out = tta.average_multiclass(probs)
      assert pytest.approx(float(out.sum()), abs=1e-6) == 1.0

  # tests/test_calibrate.py
  def test_fit_threshold_maximises_macro_f1():
      import numpy as np
      y = np.array([0, 0, 0, 1, 1, 1])
      s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
      thr = calibrate.fit_threshold(y, s, "macro_f1")
      assert 0.2 < thr < 0.8

  def test_fit_threshold_on_separable_scores_is_near_midpoint():
      import numpy as np
      y = np.array([0, 0, 1, 1])
      s = np.array([0.0, 0.1, 0.9, 1.0])
      thr = calibrate.fit_threshold(y, s, "macro_f1")
      assert 0.0 <= thr <= 1.0
      preds = (s >= thr).astype(int)
      assert (preds == y).mean() == 1.0
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_tta.py tests/test_calibrate.py -v`
  Expected: FAIL — modules missing

- [ ] **Step 3: Implement**
  - `mscocoai/tta.py`: `tta_views` as specified; `scale(x, s)` uses `torch.nn.functional.interpolate(x, size=round(img_size / s), mode="bilinear", align_corners=False)` then a centre crop of `img_size`. `average_binary` stacks 1-D tensors and means them; `average_multiclass` stacks 2-D tensors and means them.
  - `mscocoai/calibrate.py`: `fit_threshold` sweeps `grid` thresholds in `(0, 1)`, computes the objective (macro-F1 via `sklearn.metrics.f1_score(..., average="macro", zero_division=0)`, or accuracy for `"accuracy"`), and returns the best threshold (ties → the smallest). `best_threshold_from_curve` is the same computation expressed over the ROC curve; keep `fit_threshold` as the single implementation and have the curve variant delegate to it.
  - `engine`: after the final epoch (or when the budget stops the run), if `cfg.calibrate != "off"`, compute validation scores with the evaluation model and fit the threshold; store it in the checkpoint (`"threshold"`) and in `results.json` (`"threshold"`).
  - `evaluate.py`: add `--tta {none,flip,flip_scale}`; when set, accumulate probabilities across the views and average; use the checkpoint's stored `threshold` for the binary decision when present, else 0.5, and report both.
  - `ensemble.py`: add `--tta` and per-head weights (`--weight-detection`, `--weight-attribution`).
  - `train.py`: add `--tta`, `--calibrate {off,macro_f1,accuracy}`; append the fields with defaults.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_tta.py tests/test_calibrate.py -v`
  Expected: 5 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 73 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/tta.py mscocoai/calibrate.py mscocoai/evaluate.py mscocoai/ensemble.py mscocoai/engine.py mscocoai/train.py tests/
  git commit -m "feat(eval): test-time augmentation and validation-calibrated thresholds"
  ```

### Task 13: Balanced reporting protocol

**Files:**
- Modify: `mscocoai/metrics.py`, `mscocoai/evaluate.py`, `mscocoai/report.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```python
  # mscocoai/metrics.py
  def balanced_accuracy(y_true, y_pred) -> float
  def balanced_indices(y_true, seed: int = 0) -> np.ndarray   # equal count per class
  def balanced_binary_report(y_true, y_score, *, seed: int = 0, threshold: float = 0.5) -> dict
  ```
  `binary_report` gains a `"balanced_accuracy"` key.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_metrics.py`:
  ```python
  def test_balanced_accuracy_ignores_class_frequency():
      y = np.array([0] * 90 + [1] * 10)
      pred = np.zeros(100, dtype=int)
      assert metrics.balanced_accuracy(y, pred) == pytest.approx(0.5)

  def test_balanced_indices_picks_equal_counts():
      y = np.array([0] * 90 + [1] * 10)
      idx = metrics.balanced_indices(y, seed=0)
      assert len(idx) == 20
      assert (y[idx] == 0).sum() == 10 and (y[idx] == 1).sum() == 10

  def test_balanced_binary_report_reports_both_protocols():
      y = np.array([0] * 90 + [1] * 10)
      s = np.linspace(0.0, 1.0, 100)
      rep = metrics.balanced_binary_report(y, s, seed=0)
      assert "balanced" in rep and "raw" in rep
      assert rep["balanced"]["support"] == 20
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_metrics.py -v`
  Expected: FAIL — functions missing

- [ ] **Step 3: Implement**
  - `balanced_accuracy(y_true, y_pred)` = mean of per-class recall.
  - `balanced_indices(y_true, seed)`: for each class present, take `min(count)` samples without replacement using `np.random.default_rng(seed)`; return the sorted concatenated indices.
  - `balanced_binary_report(y_true, y_score, seed, threshold)`: returns `{"balanced": binary_report(y[bi], s[bi], threshold), "raw": binary_report(y, s, threshold)}`.
  - `binary_report` adds `"balanced_accuracy"`.
  - `evaluate.py`: include `result["binary_balanced"] = balanced_binary_report(...)` for the binary and multi-task tasks, and add it to `per_generator.csv` rows.
  - `report.py`: `save_per_generator_csv` gains a `balanced_accuracy` column.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_metrics.py -v`
  Expected: all pass

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 76 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/metrics.py mscocoai/evaluate.py mscocoai/report.py tests/test_metrics.py
  git commit -m "feat(metrics): balanced reporting protocol alongside raw-skew metrics"
  ```

### Task 14: Leave-one-generator-out (LOGO) protocol

**Files:**
- Modify: `mscocoai/data.py`, `mscocoai/train.py`, `mscocoai/evaluate.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: Task 5's `holdout_generator` parameter.
- Produces: `TrainConfig` gains trailing default `holdout_generator: str | None = None`. `build_dataset` already accepts `holdout_generator`; the CLI gains `--holdout-generator {sd21,sdxl,sd3,dalle3,mj6}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_data.py`:
  ```python
  def test_holdout_generator_applies_to_validation_too(tiny_dataset):
      ds = data_mod.build_dataset("validation", data_dir=str(tiny_dataset), limit=None,
                                  norm="imagenet", img_size=64, holdout_generator="sd3")
      assert 3 not in set(ds.labels_b().tolist())

  def test_holdout_generator_keeps_real_and_other_generators(tiny_dataset):
      ds = data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=None,
                                  norm="imagenet", img_size=64, holdout_generator="sd3")
      labels = set(ds.labels_b().tolist())
      assert 0 in labels and 3 not in labels and len(labels) == 5
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_data.py -v`
  Expected: FAIL — `build_dataset` does not accept `holdout_generator` (it was added in Task 5 for the parquet source only — extend it to the folder source)

- [ ] **Step 3: Implement**
  - `data.build_dataset`: pass `holdout_generator` to **both** `LocalFolderDataset` and `ParquetImageDataset`; add the parameter to `LocalFolderDataset.__init__` and filter `self.samples` by label-b id.
  - `train.py`: add `--holdout-generator`; append the field with a default.
  - `engine.fit`: pass `holdout_generator=cfg.holdout_generator` to both the train and validation `build_dataset` calls.
  - `evaluate.py`: add `--holdout-generator`; when set, the report labels the run as a LOGO evaluation and adds a `held_out_generator` key to `report.json`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_data.py -v`
  Expected: all pass

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 78 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/data.py mscocoai/train.py mscocoai/evaluate.py tests/test_data.py
  git commit -m "feat(protocol): leave-one-generator-out evaluation protocol"
  ```

---

# Phase C — Handover, platforms, and real runs

### Task 15: Parallel job launcher for a shared PC

**Files:**
- Create: `mscocoai/launcher.py`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Consumes: nothing new (shells out to `python -m mscocoai.train`).
- Produces:
  ```python
  @dataclass
  class Job:
      name: str
      cmd: list[str]
      out_dir: str
      gpu: int | None = None
      workers: int = 4
  def estimate_concurrency(free_vram_gb: float, per_job_vram_gb: float = 10.0) -> int
  def run_jobs(jobs: list[Job], *, max_concurrency: int = 1, log_dir: str = "logs",
               status_path: str = "jobs_status.json", dry_run: bool = False) -> int
  def load_jobs(path: str | Path) -> list[Job]
  ```
  CLI: `python -m mscocoai.launcher --jobs jobs.json [--max-concurrency N] [--dry-run]`.
  `jobs.json` is a list of `{"name", "cmd", "out_dir", "gpu", "workers"}` objects.

- [ ] **Step 1: Write the failing tests** — `tests/test_launcher.py`:
  ```python
  def test_estimate_concurrency_scales_with_vram():
      assert launcher.estimate_concurrency(4.0) == 1
      assert launcher.estimate_concurrency(40.0) >= 3

  def test_run_jobs_dry_run_does_not_execute(tmp_path):
      job = launcher.Job(name="j1", cmd=["echo", "hi"], out_dir=str(tmp_path / "o1"))
      n = launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                             status_path=str(tmp_path / "status.json"), dry_run=True)
      assert n == 1
      assert not (tmp_path / "o1").exists()

  def test_run_jobs_executes_and_writes_status(tmp_path):
      job = launcher.Job(name="j1", cmd=[sys.executable, "-c", "print('ok')"],
                          out_dir=str(tmp_path / "o1"))
      launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                         status_path=str(tmp_path / "status.json"))
      status = json.loads((tmp_path / "status.json").read_text())
      assert status["j1"]["status"] == "done"
      assert "ok" in (tmp_path / "logs" / "j1.log").read_text()

  def test_run_jobs_skips_already_completed(tmp_path):
      status = tmp_path / "status.json"
      status.write_text(json.dumps({"j1": {"status": "done"}}))
      job = launcher.Job(name="j1", cmd=[sys.executable, "-c", "raise SystemExit(1)"],
                          out_dir=str(tmp_path / "o1"))
      launcher.run_jobs([job], max_concurrency=1, log_dir=str(tmp_path / "logs"),
                         status_path=str(status))
      assert json.loads(status.read_text())["j1"]["status"] == "done"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_launcher.py -v`
  Expected: FAIL — module missing

- [ ] **Step 3: Implement**
  - `Job` dataclass as specified.
  - `estimate_concurrency(free_vram_gb, per_job_vram_gb=10.0)` → `max(1, int(free_vram_gb // per_job_vram_gb))`.
  - `run_jobs`: read the existing `status_path` (if any) into a dict; skip jobs already marked `done`; run the remaining jobs with a concurrency limit using `concurrent.futures.ThreadPoolExecutor`; for each job, set `CUDA_VISIBLE_DEVICES` to `str(job.gpu)` when `job.gpu is not None`, set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to `job.workers`, redirect stdout+stderr to `log_dir/<name>.log` (open the file, `subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)`), and record `{"status": "done"|"failed", "started_at", "finished_at", "returncode"}` in the status dict, writing the status file atomically after each change. Launch the child with `start_new_session=True` so it survives the parent's SSH session. Return the number of jobs that finished with returncode 0.
  - `load_jobs(path)`: read the JSON list and build `Job` objects (validating that each has `name`, `cmd`, `out_dir`).
  - `main()`: parse args, load jobs, compute the default concurrency from `shutil.disk_usage` / `nvidia-smi` free VRAM when `--max-concurrency` is not given, and call `run_jobs`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_launcher.py -v`
  Expected: 4 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 82 passed

- [ ] **Step 6: Commit**
  ```bash
  git add mscocoai/launcher.py tests/test_launcher.py
  git commit -m "feat(launcher): parallel job queue for shared-PC runs"
  ```

### Task 16: Result collector, quickstart, pinned requirements

**Files:**
- Create: `scripts/collect_results.py`, `scripts/quickstart.sh`, `requirements-lock.txt`
- Test: `tests/test_collect_results.py`

**Interfaces:**
- Consumes: Task 1's `results.json` schema.
- Produces:
  ```python
  # scripts/collect_results.py
  def collect(run_dirs: list[Path]) -> list[dict]
  def write_csv(rows: list[dict], path: Path) -> None
  def write_markdown(rows: list[dict], path: Path) -> None
  ```
  Each collected row: `{"run", "task", "backbone", "freq_branch", "clip_init", "epochs_done", "best_epoch", "best_val", "accuracy", "balanced_accuracy", "macro_f1", "auc", "threshold", "path"}`.

- [ ] **Step 1: Write the failing test** — `tests/test_collect_results.py`:
  ```python
  def test_collect_reads_results_json(tmp_path):
      run = tmp_path / "run1"; run.mkdir()
      (run / "results.json").write_text(json.dumps({
          "run_name": "run1", "best_epoch": 2, "best_val": 0.3, "threshold": 0.42,
          "config": {"task": "binary", "backbone": "resnet50"},
          "epochs": [{"epoch": 0, "val_metrics": {"accuracy": 0.8, "auc": 0.9}}]}))
      rows = collect.collect([run])
      assert rows and rows[0]["run"] == "run1"
      assert rows[0]["accuracy"] == pytest.approx(0.8)
      assert rows[0]["threshold"] == pytest.approx(0.42)

  def test_write_csv_and_markdown(tmp_path):
      rows = [{"run": "a", "task": "binary", "accuracy": 0.9}]
      collect.write_csv(rows, tmp_path / "r.csv")
      collect.write_markdown(rows, tmp_path / "r.md")
      assert "a" in (tmp_path / "r.csv").read_text()
      assert "| a |" in (tmp_path / "r.md").read_text()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_collect_results.py -v`
  Expected: FAIL — module missing (import `scripts.collect_results` via `importlib` from the repo root, or add `scripts/__init__.py`)

- [ ] **Step 3: Implement**
  - `scripts/collect_results.py`: `collect(run_dirs)` walks each directory for `results.json`, reads it, extracts the run name, config fields (`task`, `backbone`, `freq_branch`, `clip_init`), the best epoch / best val / threshold, and the metrics of the **best** epoch row (the row whose `epoch` equals `best_epoch`) from `epochs`. Skip directories without `results.json`. `write_csv` writes a header plus one row per run; `write_markdown` writes a markdown table with the same columns. CLI: `python scripts/collect_results.py <run_dir> [<run_dir> ...] --csv out.csv --md out.md`.
  - `scripts/quickstart.sh`: detect the platform (`KAGGLE_*` env → Kaggle; `/content` exists → Colab; else generic Linux), print the exact next commands (install, doctor, download, smoke train), and exit 0. Must be `bash`-compatible and must not require any secret.
  - `requirements-lock.txt`: the exact versions installed in the verification venv (`pip freeze` output of the venv, filtered to the direct dependencies), with a header comment explaining it is the reproducibility pin and that the venv is the reference environment.

- [ ] **Step 4: Run test to verify it passes**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest tests/test_collect_results.py -v`
  Expected: 2 passed

- [ ] **Step 5: Run the full suite**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q`
  Expected: 84 passed

- [ ] **Step 6: Commit**
  ```bash
  git add scripts/collect_results.py scripts/quickstart.sh requirements-lock.txt tests/test_collect_results.py
  git commit -m "feat(handover): result collector, quickstart and pinned requirements"
  ```

### Task 17: Documentation set (per-platform guides + team handover)

**Files:**
- Modify: `README.md`, `kaggle/KAGGLE_GUIDE.md`, `lab_pc/A4000_SETUP.md`
- Create: `colab/COLAB_GUIDE.md`, `colab/mscocoai_colab.ipynb`, `TEAM_GUIDE.md`, `scripts/run_ablation.sh`, `scripts/train_ddp.sh`

**Interfaces:**
- Consumes: every command produced by Tasks 1–16.
- Produces: the documented command set. No code changes beyond the two shell scripts.

- [ ] **Step 1: Write the shell scripts**
  - `scripts/train_ddp.sh`: `torchrun --nproc_per_node="${NPROC:-2}" -m mscocoai.train "$@"` with a comment on the free-tier T4×2 setting.
  - `scripts/run_ablation.sh`: two modes — `--scaled` (adds `--limit 5000 --epochs 8`) and the full grid (30 epochs, early-stop 5). Each row is a `python -m mscocoai.train …` invocation with a distinct `--out-dir outputs/<row>`; rows 1–2 use the paper protocol (ResNet-50 binary / attribution, no balance flags), rows 3–7 add the components in order. The script prints each command before running it and stops on the first failure unless `--keep-going` is passed.
- [ ] **Step 2: Verify the scripts parse**
  Run: `bash -n scripts/run_ablation.sh && bash -n scripts/train_ddp.sh && bash -n scripts/quickstart.sh`
  Expected: exit 0, no output.
- [ ] **Step 3: Rewrite `README.md`**
  New sections: the six requirements and where each is implemented; the repo-layout table including every new file; the platform matrix (Kaggle / Colab / shared A4000 / `100.123.105.15` JupyterLab); the **results table** (populated from `scripts/collect_results.py`, showing "pending" for runs not yet executed, never a projection); the paper↔code map for O1–O5 including the new components; the balanced-protocol note explaining why 0.80 must be compared on a balanced subset; the honest notes (no SOTA claim until measured; TPU untested; Colab/Kaggle-only paths unverified).
- [ ] **Step 4: Rewrite `kaggle/KAGGLE_GUIDE.md`**
  The free-tier flow: enable Internet → `!pip install -q -r requirements.txt` cell → run `mscocoai.doctor` → Session 1: download the dataset with `python -m mscocoai.download --workers 8`, then **Save Version → push to a Kaggle Dataset** → Session 2+: attach the Dataset and run with `--data-dir /kaggle/input/<dataset> --data-layout parquet`; the secrets cell (Kaggle Secrets → `MSCOCOAI_APPS_SCRIPT_URL` / `MSCOCOAI_APPS_SCRIPT_TOKEN`); disk plan (`HF_HOME=/kaggle/temp/hf`); the time budget (`--max-hours 11`); the CPU-only recipe; the TPU recipe (marked untested); the resume runbook; and the two-GPU DDP command.
- [ ] **Step 5: Write `colab/COLAB_GUIDE.md` and `colab/mscocoai_colab.ipynb`**
  The notebook must be a real `.ipynb` (JSON, 3 cells: install + secrets setup; doctor + download; train + Drive upload) that runs top-to-bottom on Colab. The guide covers: the free T4, `/content/drive` as the Drive backend (`--drive-dir /content/drive/MyDrive/mscocoai`), storing the **parquet shards** in Drive (not 96k small files) and reading them with `--data-layout parquet`, runtime-disconnect recovery, and `--max-hours`.
- [ ] **Step 6: Rewrite `lab_pc/A4000_SETUP.md`**
  The `ssh -l fast 100.123.105.15` flow: create `~/venvs/mscocoai`, install the CUDA torch build, `rsync -a --exclude .git` the repo, run `mscocoai.doctor`, download the dataset with the resumable cache, run the scaled ablation grid, launch the full grid with `nohup`/`mscocoai.launcher`, and read results back with `scripts/collect_results.py`. Include the JupyterLab URL `http://100.123.105.15:8888`.
- [ ] **Step 7: Write `TEAM_GUIDE.md`**
  The platform matrix; the per-platform one-command quickstart; what each member fills in (secrets, run name) and what they hand back (`results.json`, `summary.md`, `report/`); how to run two jobs in parallel on the shared PC via `mscocoai.launcher`; the naming convention for run directories; and the "do not commit outputs or secrets" rule.
- [ ] **Step 8: Verify every documented command exists**
  Run: for each command string extracted from the guides, check the referenced module has a `--help` that succeeds:
  ```bash
  for m in train evaluate ensemble download doctor launcher drive_upload clip_probe; do
    /tmp/opencode/venv-mscocoai/bin/python -m mscocoai.$m --help > /dev/null || echo "FAIL $m"
  done
  ```
  Expected: no `FAIL` lines.
- [ ] **Step 9: Commit**
  ```bash
  git add README.md kaggle/KAGGLE_GUIDE.md lab_pc/A4000_SETUP.md colab/ TEAM_GUIDE.md scripts/
  git commit -m "docs: per-platform guides, team handover and ablation scripts"
  ```

### Task 18: Verify on the A4000 lab server and launch real runs

**Files:**
- Create: `lab_pc/RUN_LOG.md` (the verification transcript and the launch commands actually used)

**Interfaces:**
- Consumes: every command from Tasks 1–17.
- Produces: a verified end-to-end run on the real hardware, and the real numbers for the README results table.

- [ ] **Step 1: Provision the environment on the server**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'python3 -m venv ~/venvs/mscocoai && ~/venvs/mscocoai/bin/pip -q install --upgrade pip && \
     ~/venvs/mscocoai/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cu124 && \
     ~/venvs/mscocoai/bin/pip -q install -r requirements.txt && \
     ~/venvs/mscocoai/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
  ```
  Expected: prints a torch version and `True`.
- [ ] **Step 2: Sync the repo and run the preflight**
  ```bash
  rsync -a --exclude .git --exclude __pycache__ ./ fast@100.123.105.15:~/mscocoai/
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && ~/venvs/mscocoai/bin/python -m mscocoai.doctor'
  ```
  Expected: the report shows the RTX A4000, ~109 GB free disk, and no blockers.
- [ ] **Step 3: Run the test suite on the server (GPU present)**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && ~/venvs/mscocoai/bin/python -m pytest -q'
  ```
  Expected: all tests pass.
- [ ] **Step 4: Download the dataset (resumable) and export the mirror**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && nohup ~/venvs/mscocoai/bin/python -m mscocoai.download --workers 8 \
       --cache-dir ~/hf_cache --export-images ~/mscocoai/data_mirror \
       > ~/mscocoai/logs_download.txt 2>&1 &'
  ```
  Expected: the download completes (~7.5 GB) and the mirror contains `train|validation|test` × 6 class folders.
- [ ] **Step 5: Smoke train on the GPU**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && ~/venvs/mscocoai/bin/python -m mscocoai.train --task multitask \
       --backbone resnet18 --data-dir ~/mscocoai/data_mirror --data-layout folder \
       --img-size 224 --epochs 2 --limit 2000 --batch-size 32 --num-workers 8 \
       --out-dir ~/mscocoai/outputs/smoke --drive-backend none --max-hours 0.5'
  ```
  Expected: two epochs complete, `results.json` and `summary.md` exist, and `stop_reason` is `completed`.
- [ ] **Step 6: Run the scaled ablation grid**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && nohup bash scripts/run_ablation.sh --scaled > ~/mscocoai/ablation_scaled.log 2>&1 &'
  ```
  Expected: each row writes `outputs/<row>/results.json`; collect them with `scripts/collect_results.py`.
- [ ] **Step 7: Launch the full grid in the background**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && nohup bash scripts/run_ablation.sh > ~/mscocoai/ablation_full.log 2>&1 &'
  ```
  Expected: the process is running (`pgrep -f run_ablation`). It runs for hours; that is intended — the rows are read back as they land.
- [ ] **Step 8: Record the verification in `lab_pc/RUN_LOG.md`**
  Record the exact commands, their outputs, the versions installed, the measured epoch times, and the collected results table. Mark anything that could not be executed (TPU, Kaggle Dataset attachment, Colab runtime) as **unverified**.
- [ ] **Step 9: Update the README results table from the collected results**
  ```bash
  ssh -o BatchMode=yes -l fast 100.123.105.15 \
    'cd ~/mscocoai && ~/venvs/mscocoai/bin/python scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md'
  rsync -a fast@100.123.105.15:~/mscocoai/results/ ./results/
  ```
  Paste the table into `README.md`; rows that have not finished stay "pending".
- [ ] **Step 10: Commit**
  ```bash
  git add lab_pc/RUN_LOG.md results/ README.md
  git commit -m "docs(results): verified A4000 run log and first measured results"
  ```

### Task 19: Final review and handover package

**Files:**
- Modify: `README.md` (only if the review finds gaps)
- Create: `HANDOVER.md` (a one-page checklist for the team)

**Interfaces:**
- Consumes: everything above.
- Produces: a handover package that a team member can use without asking questions.

- [ ] **Step 1: Run the full verification suite one more time**
  Run: `/tmp/opencode/venv-mscocoai/bin/python -m pytest -q` and `/tmp/opencode/venv-mscocoai/bin/python -m py_compile mscocoai/*.py`
  Expected: all tests pass, no compile errors.
- [ ] **Step 2: Verify the working tree is clean of artefacts**
  Run: `git status --porcelain | grep -E '\.pt$|\.npy$|__pycache__|\.pytest_cache|outputs/'` and confirm no matches; also confirm no secret values appear anywhere: `grep -rIn "Papa@404\|apps_script_token.*=" --exclude-dir=.git . | grep -v "MSCOCOAI_APPS_SCRIPT_TOKEN"` returns nothing.
- [ ] **Step 3: Write `HANDOVER.md`**
  One page: the four platforms and their one-command entry points; the three things a member must configure (secrets, run name, output location); the three things a member must return (`results.json`, `summary.md`, `report/`); how to run in parallel on the shared PC; where the real results live; who to contact for what; and the honest list of unverified paths (TPU, Kaggle Dataset attachment, Colab runtime).
- [ ] **Step 4: Final diff review**
  Run: `git diff --stat HEAD~1` and read the diff for accidental secrets, debug prints, or unrelated changes.
  Expected: only intended files changed.
- [ ] **Step 5: Commit**
  ```bash
  git add HANDOVER.md README.md
  git commit -m "docs: handover package"
  ```
- [ ] **Step 6: Report to the user**
  Summarise: what was implemented, the measured numbers (or "pending"), the verification evidence (test counts, server run), and the explicit list of unverified paths.

---

## Appendix — Command reference produced by this plan

```bash
# preflight
python -m mscocoai.doctor

# dataset
python -m mscocoai.download --workers 8 --export-images ./data_mirror
python -m mscocoai.download --verify

# paper protocol (unchanged)
bash scripts/run_paper_models.sh

# SOTA profile
python -m mscocoai.train --task multitask --backbone convnext_tiny --freq-branch dct \
    --ema-decay 0.999 --label-smoothing 0.05 --tta flip_scale --calibrate macro_f1 \
    --data-dir ./data_mirror --data-layout folder --out-dir outputs/mt_freq

# CLIP branch
python -m mscocoai.train --task multitask --clip-init clip_vitl14 --grad-checkpointing \
    --data-dir ./data_mirror --out-dir outputs/mt_clip

# LOGO
python -m mscocoai.train --task binary --backbone resnet50 --holdout-generator sd3 \
    --data-dir ./data_mirror --out-dir outputs/logo_sd3

# multi-GPU
bash scripts/train_ddp.sh --task binary --backbone resnet50 --data-dir ./data_mirror \
    --out-dir outputs/ddp

# TPU (untested on real hardware)
bash scripts/train_tpu.sh --task binary --backbone resnet50 --data-dir ./data_mirror \
    --out-dir outputs/tpu

# parallel shared-PC jobs
python -m mscocoai.launcher --jobs jobs.json --max-concurrency 2

# aggregate
python scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md
```
