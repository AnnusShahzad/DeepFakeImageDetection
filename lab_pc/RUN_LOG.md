# MS-COCOAI — A4000 Lab Server Verification Log

Host: `LAB12-18` (`ssh -l fast 100.123.105.15`), RTX A4000 16 GB, 24 vCPU,
31 GB RAM, ~97 GB free disk. Python 3.12.3.

## What was verified

| Check | Command | Result |
|---|---|---|
| SSH (key auth) | `ssh -o BatchMode=yes -l fast 100.123.105.15` | OK |
| Environment preflight | `python -m mscocoai.doctor` | OK — A4000 15.6 GB VRAM, 28.3 GB avail RAM, 96.8 GB free, HF reachable, no blockers |
| Unit tests | `python -m pytest -q` | **105 passed** |
| Editable install | `pip install -e .` | OK (`python -m mscocoai.*` importable from any cwd) |
| Dataset download | `python -m mscocoai.download --workers 8 --cache-dir ~/hf_cache --export-images ~/mscocoai/data_mirror` | OK — 17 parquet shards in ~8 min; mirror export: **96,000 images**, 7.51 GB written, 0 unknown, `manifest.json` written |
| Mirror sanity | `find data_mirror -type f \| wc -l` | 96,000 |
| Synthetic end-to-end (CLI) | `mscocoai.train --task multitask --backbone resnet18 --data-dir /tmp/smoke_mirror ...` | OK — `results.json` + `summary.md` written, resume path works |
| Real end-to-end smoke | `mscocoai.train --task multitask --backbone resnet18 --data-dir ~/mscocoai/data_mirror --img-size 64 --limit 2000 --epochs 2` | OK — epoch 1 best val loss 1.79; `results.json` complete |
| Scaled ablation grid | `bash scripts/run_ablation.sh --scaled --keep-going` | **RUNNING** (see below) |
| Full ablation grid | `bash scripts/run_ablation.sh` | **not launched** — multi-day; see "What is not verified" |

## Notes / incidents

1. **Transient SSH failure** ("Host key verification failed") during one
   concurrent `rsync`. Re-verified: the server's host key fingerprint is
   **identical** to the cached one (ED25519 `SHA256:6h5hSV2YC...`) — no key
   change; the failure was transient and subsequent connections work. Reported
   honestly for the record.
2. **First CLI run failed**: the parser rewrite had dropped
   `--lambda-d / --lambda-a / --val-metrics-each-epoch`, which `TrainConfig`
   requires. Fixed; added `tests/test_train.py` regression test; suite is green.
3. CUDA torch build installed: `torch 2.15.0+cu132` (cu124 wheel was replaced by
   the resolver with cu132 — fine for the installed driver 595.99).

## Data layout used for verification

```
~/mscocoai/data_mirror/{train,validation,test}/{real,sd21,sdxl,sd3,dalle3,mj6}/<shard>_<row>.<ext>
```

Parquet dump also kept at `~/hf_cache/datasets--Rajarshi-Roy-research--Defactify_Image_Dataset`
for `--data-layout parquet` runs (used on Kaggle-like flows).

## Scaled ablation grid

Command: `DATA_DIR=~/mscocoai/data_mirror MSCOCOAI_PYTHON=~/venvs/mscocoai/bin/python bash scripts/run_ablation.sh --scaled --keep-going`

Log: `~/mscocoai/ablation_scaled.log`; per-row logs: `~/mscocoai/outputs/ablation/<row>.log`;
result aggregation (after completion): `python scripts/collect_results.py ~/mscocoai/outputs/* --csv results/ablation.csv --md results/ablation.md`

Rows (8 epochs each, 5,000 imgs/split): paper_resnet50_binary, paper_resnet50_attr,
mt_convnext_tiny, mt_freq, mt_freq_ema_ls, mt_freq_ema_ls_tta, mt_clip.

**Status: in progress.** Numbers will be added to this file and `README.md` when the grid finishes.

## What is not verified here

- **TPU** (`--device tpu`, `torch_xla`): written, **untested on real TPU hardware** — none exists.
- **Kaggle Dataset attachment** (Save Version → Promote to Dataset) and the
  **Colab runtime**: user-level Kaggle/Colab actions; guides mark them **unverified**.
- **Full 30-epoch grid**: would take days on the shared PC; not launched without approval.
- Test split is read **only** at final evaluation (never used for selection).
