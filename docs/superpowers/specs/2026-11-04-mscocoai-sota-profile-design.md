# MS-COCOAI: SOTA Profile, Results Persistence, Free-Tier & Multi-Device Pipeline — Design

Date: 2026-11-04
Status: **awaiting approval**
Extends: `docs/superpowers/specs/2026-11-04-mscocoai-training-pipeline-design.md` (base pipeline, already implemented)
Owner: Muhammad Annus Shahzad (with project "DLP — OBJECTIVE.md")

## Intent

Close the gap between the base pipeline (which reproduces O1–O5 faithfully) and a
pipeline that (a) never loses work, (b) runs unattended on Kaggle's free tier and on
a shared lab PC, (c) uses every available device, and (d) has a credible shot at
beating the published MS-COCOAI baselines — **0.80** binary detection accuracy and
**0.45** attribution accuracy — with a method that is novel *relative to that
baseline* and defensible in review.

The honest framing, carried into the README: no code makes a paper "automatically
accepted". What this design buys is a method whose novelty is measurable (an
ablation grid over independently switchable components), an evaluation protocol
stronger than the baseline's, and real measured numbers instead of claims.

## Goals

- **G1 (results persistence).** Training writes a durable, resumable *result log* —
  not just weights — so a crash, power loss, or Kaggle session kill never loses the
  metrics already paid for. The log survives restart, is human-readable, and is
  pushed to Google Drive alongside the checkpoints.
- **G2 (Kaggle free tier).** `pip install` → dataset → train → Drive backup works
  unattended on a free Kaggle session (T4/P100, CPU-only, or TPU), inside the disk
  quota, inside the session time limit, with no admin rights and no secret in git.
- **G3 (dataset download).** A parallel, resumable, verifiable dataset fetch that
  saturates the link, survives interruption, and optionally materialises a
  folder mirror that makes the training dataloader non-blocking.
- **G4 (SOTA profile).** One codebase, independently switchable components, each
  ablatable: frequency-aware dual-branch multi-task, strong backbone, CLIP
  fine-tune branch, test-time augmentation + validation calibration, checkpoint
  ensembling, plus a leave-one-generator-out (LOGO) generalisation protocol.
- **G5 (devices).** Multi-GPU via `torchrun`/DDP, multi-TPU via `torch_xla`, single
  GPU, and CPU — one entry point, auto-selected by default, no dataloader
  bottleneck.
- **G6 (real runs).** The A4000 lab server actually runs the code end to end and
  produces the numbers the paper quotes.
- **G7 (handover).** A team member with no prior context can run the pipeline on
  Kaggle, Colab, or a shared PC — including several runs in parallel on one
  machine — from one quickstart command, and hand back results in a form the
  lead can aggregate without touching the code.

## Non-goals / anti-goals

- No re-splitting or re-labelling of MS-COCOAI; official splits only; test read
  only *after* training and model selection are finished.
- No Google Cloud project, service account, OAuth client, or API key — the existing
  constraint stands.
- No hosted serving stack, no dataset redistribution (research-use licence).
- No claim of SOTA before the numbers exist in `results/`.
- No "TPU support" that is a README paragraph and no code: if a TPU path ships, it
  ships as a real code path — and it is labelled *untested on TPU hardware*,
  because this environment has no TPU and pretending otherwise would be a lie.

## Constraints (measured, not assumed)

| Fact | Value | Source |
|---|---|---|
| Lab server | `LAB12-18`, RTX A4000 16 GB, 24 vCPU, 31 GB RAM, 109 GB free | `ssh -l fast 100.123.105.15` |
| Lab server python | 3.12.3, **no torch installed** | `python3 -c "import torch"` |
| Dataset repo | 17 parquet shards under `data/`, ~7.5 GB | HF tree API |
| Kaggle free tier | T4/P100 16 GB, 9–12 h session, ~57 GB `/kaggle/working`, CPU-only tier available | Kaggle docs |
| Baselines to beat | 0.80 detection ACC, 0.45 attribution ACC | OBJECTIVE.md §I.E |
| Label skew | Label_A 1:5 real:fake (16k real / 80k fake); Label_B balanced 16k×6 | dataset card |

## A finding that changes the headline metric

Label_A is 1:5, so the 45,000-image test split is ≈7,500 real / 37,500 fake. A
classifier that always answers "AI-generated" therefore scores **83.3 %** accuracy
on the raw test split — *above* the published 0.80 baseline. OBJECTIVE.md §II.A
says the reviewed studies (including the MS-COCOAI baseline paper) report ACC "on
balanced test sets", so 0.80 was measured on a class-balanced subset; 0.80 on the
raw skewed split is not a meaningful number.

Consequences, adopted as the reporting protocol (§8): every model reports the full
metric set, and the headline comparison against 0.80 / 0.45 is made on the
**balanced** protocol — a class-balanced test subset (7,500 real + 7,500 fake,
seeded) — with raw-skew ACC reported next to it and never alone. Attribution is
unaffected (Label_B is balanced 16k per class), so 0.45 is directly comparable.

## Approaches considered

The only genuinely open fork was **what "the SOTA core" is**. Three were put to the
user; **C was chosen**.

1. **A — Frequency-aware dual-branch only.** RGB + DCT spectrum branch, two heads.
   Cheapest, fits 16 GB at batch 32, cleanest "we improve on the paper's own
   frequency-domain ResNet-50" story.
2. **B — Foundation-model ViT-L/14 + TTA/ensemble only.** CLIP-initialised multi-task
   fine-tune. Follows the UniFD direction the literature review calls strongest, but
   costs ~2–3× per epoch and needs gradient checkpointing on 16 GB.
3. **C — Composable profile (chosen).** A + B + TTA + calibration + ensembling +
   LOGO, each behind its own flag, so the same code is both the best model *and* the
   paper's ablation grid. A stays individually selectable and remains the headline
   contribution; B is an optional branch.

Other approaches rejected along the way:

- *Mid-training checkpoint averaging (SWA)* — rejected: EMA already provides weight
  averaging and is implemented; two averaging mechanisms would muddy the ablation.
- *Diffusion-reconstruction features (DIRE)* — rejected: two full diffusion forward
  passes per test image (~100× inference cost) and diffusion-only — it would not
  cover the DALL-E 3 / MJ v6 rows that make attribution hard.
- *Re-splitting the benchmark for cross-validation* — rejected: OBJECTIVE.md §III.C
  forbids re-splitting. LOGO holds out one *generator* from training only, so the
  official splits are untouched.

## What gets built

### 1. Result persistence — `mscocoai/results.py` (new)

A `ResultsLog` owns four artefacts in the run directory, all written atomically
(tmp file → `fsync` → `os.replace`, except the append-only log):

| File | Content | Write policy |
|---|---|---|
| `history.jsonl` | one JSON object per line: epoch rows, partial rows, stop reasons | append-only; flushed + `fsync` per row |
| `metrics.csv` | flat table of the epoch rows | rewritten atomically from the whole history each epoch |
| `results.json` | run metadata + config + best epoch/loss + full epoch list | atomically replaced each epoch |
| `summary.md` | markdown table for the paper | atomically replaced each epoch |

**Keying and resume.** Rows are keyed, not blindly appended: an epoch row is keyed by
`epoch`, a partial row by `(epoch, step)`. Writing a row that already exists
**replaces** it. This makes the log idempotent under re-runs and removes the
duplicate-epoch failure mode entirely. The log is parsed at open time with a
tolerant reader that drops a truncated final line (the classic crash-during-append
artefact) and keeps everything before it.

**Source of truth.** The *checkpoint* is the ceiling for resuming (it carries the
optimizer, scheduler, scaler and EMA); the *log* is the ceiling for reporting. The
training loop writes the checkpoint first and then the epoch row, so the invariant
`log_epoch >= checkpoint_epoch` holds at all times, and a re-run of an interrupted
epoch simply upserts its row. `start_epoch = checkpoint.epoch + 1`.

**Mid-epoch resume.** The checkpoint schema gains `step_in_epoch` and `sampler_epoch`.
On resume, the training loop skips the first `step_in_epoch` batches of the epoch.
This is *approximate* — worker RNG state for augmentation is not restored, so the
skipped batches differ from the original ones — and it is documented as such. The
alternative (restart the epoch) is also available via `--resume-epoch-only`.

**Partial rows.** Every `--save-steps` optimizer steps the log records
`{"type": "partial", "epoch", "step", "train_loss"}`, so an interrupted epoch is
visible rather than vanished.

**Time budget.** `--max-hours` is checked **inside** the batch loop, not only
between epochs: on exceed, the engine writes a partial checkpoint + partial row,
saves, uploads, writes `summary.md` with `stop_reason: "time_budget"`, and exits
cleanly. A Kaggle session limit is a hard kill, so an epoch-boundary-only check
would lose up to an epoch of work.

**Drive upload.** `_maybe_upload` gains `--upload-metrics` (default on), pushing
`history.jsonl`, `metrics.csv` and `results.json` next to the weights.

### 2. Dataset download and data sources — `mscocoai/download.py` (new)

`python -m mscocoai.download [--workers N] [--cache-dir DIR] [--export-images DIR] [--verify]`

- `huggingface_hub.snapshot_download(repo_type="dataset", allow_patterns=["data/*.parquet","README.md"], max_workers=N)`
  → parallel file downloads. The HF cache keeps `*.incomplete` partials, so an
  interrupted run **resumes** rather than restarting.
- `HF_HUB_ENABLE_HF_TRANSFER=1` is set before `huggingface_hub` is imported when
  `hf_transfer` is installed → chunk-parallel transfer inside each file, which is
  what actually saturates a fast link (16 shards in parallel is not enough).
- `--export-images DIR` materialises the folder mirror
  (`DIR/<split>/<real|sd21|sdxl|sd3|dalle3|mj6>/<n>.<ext>`) from the parquet
  shards with a process pool. Image bytes are written **verbatim** (no decode) with
  the extension sniffed from the magic bytes — fast, lossless, and it turns the
  dataloader into plain JPEG reads off NVMe instead of parquet row decoding.
  A `manifest.json` makes the export resumable; existing files are skipped.
- `--verify` re-checks every shard's size against the repo API.
- Progress prints percent, MiB/s and ETA.

**Three data sources, one contract.** `mscocoai/data.py` gains a third source so a
Kaggle Dataset of parquet shards can be read directly:

| Source | Flag | Used for |
|---|---|---|
| HF hub (default) | *(none)* | first download |
| Folder mirror | `--data-dir DIR` | fastest dataloader; local mirror on the lab PC |
| **Parquet directory** | `--data-dir DIR --data-layout parquet` | a Kaggle Dataset attached as input — no export step needed |

All three yield the same sample dict (`image`, `label_a`, `label_b`, `index`).

### 3. Kaggle free tier — `mscocoai/doctor.py` (new) + CLI flags

`python -m mscocoai.doctor` reports python/torch/CUDA, GPU name + VRAM, RAM, free
disk on `/` and on the working dir, internet reachability (HF + the Drive upload
endpoint), dataset presence, output-dir writability, and prints a verdict list.
Exit code 1 if a blocker is found. This is the "run this first" command on any new
machine.

New `train.py` flags:

| Flag | Default | Purpose |
|---|---|---|
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `cuda:N` / `tpu` |
| `--num-workers` | `auto` | resolves to `min(8, max(2, cpu//2))`; `0` only when `cpu_count <= 2` |
| `--max-hours` | `0` (off) | graceful stop; `11.0` in the Kaggle script |
| `--min-free-gb` | `5` | refuse to start below this free-disk threshold |
| `--precision` | `auto` | `fp16` (Turing/Ampere) / `bf16` (Ampere+) / `fp32` |

Kaggle-specific flow, documented in `kaggle/KAGGLE_GUIDE.md`:

1. **Do not re-download 7.5 GB every session.** `/kaggle/working` and
   `/kaggle/temp` are both wiped at session end, but **Save Version → push to a
   Kaggle Dataset** persists the parquet files, and a Dataset can be attached as an
   input to every later session. Creating a Dataset is a user-level action — no
   admin, no Google Cloud. Session 1 downloads; session 2+ attaches the Dataset and
   starts training in seconds. Because the parquet directory is a first-class data
   source (§2), no export step is needed in between.
2. `HF_HOME=/kaggle/temp/hf`, `TORCH_HOME=/kaggle/temp/torch` keep the cache off the
   working-dir quota; only final artefacts are copied to `/kaggle/working`.
3. Secrets: the Apps Script URL + token come from **Kaggle Secrets** (added as
   notebook secrets), read via `os.environ` — never written to a file, never
   printed.
4. CPU-only free tier: `--device cpu --batch-size 8 --epochs 5` runs end to end.
   Note the cost: a ResNet-50 epoch over 42k images is ≈3 min on a T4 but
   ≈1–1.5 h on Kaggle's 4-vCPU tier, so the CPU recipe is a smoke test, not a
   paper run.
5. TPU: `--device tpu` under `torch_xla` (see §4).

### 4. Devices — `mscocoai/distributed.py` (new) + `mscocoai/xla.py` (new)

- **DDP.** `torchrun --nproc_per_node=N -m mscocoai.train …`. `init_distributed()`
  reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK`, picks `nccl` on CUDA and `gloo`
  otherwise, and pins the device. The train loader uses `DistributedSampler`
  (`set_epoch` each epoch); the model is wrapped in `DistributedDataParallel`; the
  validation pass gathers scores and labels across ranks so the reported metrics are
  computed once, over the whole split, on rank 0.
- **Rank discipline.** Only rank 0 evaluates, writes the result log, saves
  checkpoints and uploads to Drive; the other ranks wait at a barrier. A rank-0-only
  write path is what keeps the result log valid under DDP.
- **TPU.** `--device tpu` activates `torch_xla`: `xm.xla_device()`, XLA-aware
  seeding, `xm.mark_step()` instead of a CUDA sync, no `GradScaler` (XLA uses bf16
  autocast). `scripts/train_tpu.sh` launches it on Kaggle's TPU v3-8.
  **Labelled untested on real TPU hardware** — this environment has none; the code
  path is written and import-checked, not benchmarked.
- **No dataloader bottleneck.** `persistent_workers=True`, `prefetch_factor=4`,
  `pin_memory=True`, `non_blocking=True` transfers, JPEG decoding on N workers,
  and an optional `--channels-last` memory format for the tensor-core GPUs.

### 5. SOTA profile — the composable core

The existing `--task` (`binary` / `attribution` / `multitask`) and `--backbone`
(timm name) stay exactly as they are: they are the paper-faithful axes, and
`scripts/run_paper_models.sh` keeps working untouched. The new components are
**orthogonal flags**, not a third overlapping axis:

| Flag | Values | Component |
|---|---|---|
| `--freq-branch` | `none` / `dct` / `highpass` / `both` | frequency-aware dual-branch (§5.1) |
| `--clip-init` | `none` / `clip_vitb16` / `clip_vitl14` | foundation-model backbone (§5.2) |
| `--tta` | `none` / `flip` / `flip_scale` | test-time augmentation (§5.3) |
| `--calibrate` | `off` / `macro_f1` | validation-fitted decision threshold (§5.4) |
| `--holdout-generator` | *(none)* / `sd21`…`mj6` | LOGO protocol (§5.5) |

`--task multitask --freq-branch dct --clip-init none` is the frequency-aware
dual-branch model; `--task multitask --freq-branch dct --clip-init clip_vitl14` is
the full profile. Every combination is trainable and every combination is one row of
the ablation table.

**5.1 Frequency branch — `mscocoai/frequency.py`.** A separable 2D DCT-II basis
(precomputed matrices, applied with two `matmul`s) on the grayscale of the image;
the lowest `K×K` coefficients are kept (`--freq-k`, default 32), log-scaled, and fed
to a small CNN; the branch output is concatenated with the backbone feature vector.
`--freq-mode` selects DCT, high-pass (image − Gaussian blur, exposing the upsampling
residual the frequency literature attributes to generator fingerprints), or both.

*Pixel-domain correctness.* The branch must consume true 0..1 pixel values, not the
normalised tensor (normalisation subtracts a per-channel mean, which destroys the
DC term and makes the "grayscale" meaningless). Rather than change the data
contract, the branch **de-normalises** its input using the mean/std recorded in the
checkpoint config — mathematically exact, zero change to the paper path.

**5.2 CLIP branch.** `--clip-init clip_vitl14` (or `clip_vitb16`) builds the
multi-task model on a CLIP ViT encoder with the same two heads, fine-tuned with the
paper's optimizer settings. Gradient checkpointing is available for 16 GB cards.
`mscocoai/clip_probe.py` is **not** replaced: it remains the frozen-linear-probe
path that OBJECTIVE.md O2 specifies, and the new branch is the *fine-tuned*
counterpart.

**5.3 TTA — `mscocoai/tta.py`.** Deterministic test-time augmentation: identity,
horizontal flip, and two centre-crop scales (1.0, 0.875). Probabilities are
averaged. Evaluation only, never in training.

**5.4 Calibration — `mscocoai/calibrate.py`.** The binary decision threshold is
fitted on the **validation** split (never test) to maximise macro-F1, and stored in
the checkpoint. `evaluate.py` uses the stored threshold when present and reports the
0.5-threshold numbers alongside it.

**5.5 LOGO protocol.** `--holdout-generator sd21` removes that generator from
**training and validation** (validation too — early stopping must not see the
unseen generator). It is then scored on the test split, *after* training has
finished and all selection decisions are closed:
- binary detection of real vs. the held-out generator (ACC, AUC, F1) — the
  unseen-generator number;
- attribution accuracy among the four *seen* fake generators — a sanity check that
  the model did not collapse.

Test is therefore never used for any training or selection decision; each LOGO run
produces one final test report, which does not violate the protocol.

**5.6 Self-describing checkpoints.** The checkpoint gains a `model_spec` blob
(`task`, `backbone`, `freq_branch`, `clip_init`, `freq_k`, `img_size`, `norm`,
`mean`, `std`). `load_model_for_eval`, `evaluate.py` and `ensemble.py` rebuild from
`model_spec`, falling back to the legacy `task`/`backbone` fields for checkpoints
written before this change. A checkpoint must never load into a silently different
network.

**5.7 Ensembling.** `mscocoai/ensemble.py` is extended to mix architectures, apply
per-head weights, and combine with TTA.

### 6. Ablation grid and compute budget

Two grids, both produced by the same code that produces the model.

**Scaled grid (first, ~1 h on the A4000).** `--limit 5000 --epochs 8`, one row per
component. Purpose: prove each component trains and helps, and catch a broken
component before spending GPU-days.

**Full grid (background, the paper's tables).** 30 epochs, early-stop 5, on the
full 42k train split:

| # | Configuration | Purpose in the paper |
|---|---|---|
| 1 | ResNet-50 binary (paper protocol) | O1 reproduction, reference point |
| 2 | ResNet-50 attribution (paper protocol) | O3 reproduction, reference point |
| 3 | ConvNeXt-Tiny multitask | strong single backbone |
| 4 | 3 + `--freq-branch dct` | **frequency-aware dual-branch** |
| 5 | 4 + EMA 0.999 + label smoothing 0.05 | training-schedule ablation |
| 6 | 5 + `--tta flip_scale --calibrate macro_f1` | inference-time ablation |
| 7 | 6 + `--clip-init clip_vitl14` | foundation-model branch |
| 8 | ensemble of 4 + 5 (+7) | final reported model |

Rows 4–8 are the ablation; row 8 is the headline. Estimated ≈3 h/row for ConvNeXt
rows and ≈8 h/row for the ViT-L row on the A4000 — the grid is a multi-day job on a
shared PC and is launched with `nohup`, one row at a time, with results read back
from each run's `results.json`. Rows are added as they land; the README table shows
real numbers or "pending", never a projection.

### 7. Real runs on the lab server

- Create `~/venvs/mscocoai` on `100.123.105.15`, install the CUDA torch build +
  deps, rsync the repo (excluding `.git`).
- Run `mscocoai.doctor`, then the downloader (background, resumable cache), then a
  smoke train (`--limit 200 --epochs 2`), then the real runs.
- Real runs are launched with `nohup` so they survive the SSH session; their
  `history.jsonl` / `results.json` are read back for the README results table.
- The JupyterLab server on the same box (`:8888`) is the interactive front end for
  the same files.

### 8. Metric reporting protocol

Every final report carries, for the binary task: accuracy (raw-skew **and**
balanced-subset), balanced accuracy, macro-F1, F1, AUC, PR-AUC, FPR@95 %TPR,
precision, recall — plus per-generator breakdowns. For attribution: accuracy,
macro-F1, per-class P/R/F1, the 6×6 confusion matrix (raw and row-normalised),
and the balanced-subset accuracy (identical to raw, since Label_B is balanced).

The headline comparison table uses the **balanced** protocol for detection and raw
accuracy for attribution, with a footnote stating the protocol. `metrics.py` gains
`balanced_binary_report(y_true, y_score, seed)` which builds a class-balanced
subset (7,500 real + 7,500 fake, seeded) and reports on it. The balanced subset is
constructed from the **test** split only at final-report time, and from
**validation** when calibrating.

### 9. Docs and tests

- `README.md` — new repo layout, new commands, an honest results table (filled from
  real runs, "pending" until then), the paper↔code map for O1–O5, and the
  balanced-protocol note.
- `kaggle/KAGGLE_GUIDE.md` — rewritten for the free tier: the Dataset-attachment
  trick, `!pip install` cell, secrets, disk plan, time budget, CPU fallback, TPU
  section, resume runbook.
- `lab_pc/A4000_SETUP.md` — the `fast@100.123.105.15` SSH flow, venv creation,
  rsync, `nohup` runs, JupyterLab URL.
- New tests: result-log crash-truncation recovery, upsert/no-duplicate-epoch,
  resume-with-existing-history, downloader against a local fake repo, DCT against a
  reference implementation, de-normalisation exactness, TTA determinism, calibration
  behaviour, balanced-report correctness, DDP helpers under a single-process gloo
  group, model factory shapes for every component combination.

### 10. Multi-platform handover (added at approval)

The deliverable must be handed to team members who will run it on **Kaggle**,
**Colab**, or a **shared PC in parallel**. Three additions:

**10.1 Colab target.** `colab/COLAB_GUIDE.md` + `colab/mscocoai_colab.ipynb` (a
real notebook, not prose). Colab specifics: free T4, `!pip install -q -r
requirements.txt` cell, `/content/drive` mounted — so the existing `copy` Drive
backend works verbatim with `--drive-dir /content/drive/MyDrive/mscocoai`.
Colab has no "dataset" concept, so the 7.5 GB parquet is stored **in Drive as the
17 shard files** (few large files = fast) and read through the new
parquet-directory source (§2) — not as 96,000 small files, which Drive would serve
painfully slowly. Runtime disconnect is the Colab failure mode, so `--max-hours`
and the autosave are mandatory, and the guide states the resume command.

**10.2 Shared-PC parallel launcher — `mscocoai/launcher.py`.** A queue runner so
several jobs (team members' runs, or grid rows) execute concurrently on one
machine without colliding:
- each job gets its own `--out-dir`, its own log file (`logs/<job>.log`), and its
  own slot in `jobs_status.json`;
- `--max-concurrency` defaults to a VRAM-aware estimate (1 full-model job per
  16 GB of free VRAM; more for CPU-only or `--limit` smoke jobs);
- CPU workers are budgeted per job (`--workers-per-job`) against `nproc`, so N
  concurrent jobs do not each spawn 8 dataloader workers and thrash the box;
- jobs are launched with `setsid`/`nohup` so they survive the SSH session;
- the launcher is itself restartable — it re-reads `jobs_status.json` and skips
  jobs that already completed.

**10.3 Team handover.** `TEAM_GUIDE.md` at the repo root: a platform matrix
(Kaggle / Colab / shared A4000 / the `100.123.105.15` JupyterLab box), the
one-command quickstart per platform, exactly what each member must fill in
(secrets, run names) and what they must hand back (`results.json`, `summary.md`,
the `report/` directory). Plus:
- `requirements-lock.txt` — exact pinned versions for reproducibility;
- `scripts/collect_results.py` — gathers `results.json` from any number of run
  directories into one CSV + markdown table (this is what the team sends back,
  and what fills the README table);
- a single `scripts/quickstart.sh` that detects the platform (Kaggle / Colab /
  generic Linux) and prints the right commands, so a new member's first action is
  one command, not a reading exercise.

**10.4 Verification for the handover.** `scripts/quickstart.sh` must run
end-to-end on the lab server in a throwaway directory (tiny `--limit`) and on this
JupyterLab box. The Colab and Kaggle paths are verified by *executing their exact
command sequence* where possible; anything that cannot be executed here (Colab's
runtime, Kaggle's Dataset attachment, TPU) is marked **unverified** in the guides
rather than implied to work.

## Verification

1. `pytest` — the existing 36 tests plus the new ones, on CPU, no network.
2. `py_compile` on every module.
3. Crash test: start a run, `SIGKILL` it mid-epoch, restart with `--resume auto`,
   assert the history has no duplicate epochs, no truncated line, and that the run
   continues from the last intact epoch.
4. Download test: interrupt the downloader, restart, assert it resumes from the
   cache instead of re-fetching.
5. DDP test: `torchrun --nproc_per_node=2` on CPU (gloo) with a tiny dataset — two
   processes, one result log, no duplicate writes.
6. Server test: the full pipeline runs on the A4000 for real; the numbers that come
   back are the numbers in the README.
7. Kaggle test: the documented command sequence is executed as written on the lab
   server with the Kaggle paths substituted. The Kaggle-only parts (Secret
   injection, TPU runtime, Dataset attachment) are stated as **unverified** if they
   cannot be exercised here, rather than implied to work.

## Risks / open points

- **ViT-L/14 on 16 GB.** Needs gradient checkpointing and a small batch; slower per
  epoch than the CNN branch. Mitigated: the CLIP branch is optional and ablatable;
  the frequency branch is the cheap, high-value component.
- **TPU path is untested.** No TPU exists in this environment. Mitigated: the
  implementation is thin and isolated, and every claim about it is marked untested.
- **Kaggle Dataset attachment.** Creating a Dataset from a notebook is a user-level
  Kaggle action, but the *first* session still has to complete a 7.5 GB download
  within one session, and `/kaggle/temp` is wiped between sessions — so the
  downloader's intra-session resume helps only within a session. The
  Dataset-attachment route is therefore the only durable path, and the guide leads
  with it.
- **Apps Script throughput.** 24 MB base64 parts over HTTPS are slower than a
  direct Drive API write, but the API route is blocked without a Cloud project.
  Mitigated: only the small fp16 weights + result logs are uploaded.
- **Shared-PC etiquette.** The lab PC is shared; the grid runs one job at a time
  via `nohup`, and each run's checkpoint autosave means a stolen session loses at
  most one epoch.
- **Accuracy claim.** Until `results/` contains real numbers, the README says
  "pending", not "SOTA".

## What's left to the builder

Implement §1–§9 in that order (each independently verifiable), run the server
verification, and fill the README results table from the real runs. The base
pipeline's paper-faithful defaults stay untouched; every new behaviour is behind an
explicit flag, so `scripts/run_paper_models.sh` still reproduces the paper exactly.
