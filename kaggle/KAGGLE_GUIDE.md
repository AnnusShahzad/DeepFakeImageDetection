# Running MS-COCOAI on Kaggle (free tier)

One Kaggle GPU notebook per run. Everything below is copy-paste; **Internet must
be enabled** (Kaggle notebook → right sidebar → **Internet: On**) or the dataset
(~7.5 GB parquet on Hugging Face) cannot be fetched.

## Cheat sheet (free tier)

| Concern | Setting |
|---|---|
| Runtime | Accelerator: **GPU** (T4 ×2 or P100 ×... of it — free tier gives T4 ×2) |
| Time limit | 9–12 h session; use `--max-hours 11` to stop cleanly before the limit |
| Working disk | `/kaggle/working` ≈20 GB — fits one 7.5 GB download + checkpoints |
| RAM | ≈13 GB — use `--batch-size 32` for ResNet-50; `--batch-size 16` for ViT-L |
| T4 precision | no bf16 — the code automatically uses fp16 autocast |
| Persistence | `/kaggle/working` saved **only** on Save Version; use `--drive-backend apps_script` for per-epoch backup |

## 0. One-time setup

1. **New Notebook** → Accelerator: **GPU** → Internet: **On**.
2. Get the code in:

   **Option A — git clone (easiest after you push the repo):**
   ```
   !git clone https://github.com/AnnusShahzad/DeepFakeImageDetection.git /kaggle/working/DeepFakeImageDetection
   %cd /kaggle/working/DeepFakeImageDetection
   ```
   **Option B — upload as a Kaggle Dataset:** zip the `Project/` folder, create a
   dataset from it, attach it to the notebook, then unzip:
   ```
   import zipfile
   zipfile.ZipFile('/kaggle/input/<your-dataset-slug>/Project.zip').extractall('/kaggle/working/')
   ```
3. Install deps (Kaggle pre-installs torch/timm; this fills the gaps):
   ```
   !cd /kaggle/working/DeepFakeImageDetection && pip install -q -r requirements.txt
   ```

## 1. Dataset: download once, then *attach as a Kaggle Dataset*

`/kaggle/temp` is wiped at every session end, so the HF cache alone does not
survive. The fix is a user-level action — no admin, no Google Cloud:

**Session 1 (download once):**
```
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.download --workers 8 \
    --cache-dir /kaggle/working/hf_cache
```
Then **Save Version** → in the finished version's **Output** tab, click
**"Promote to Dataset"** → give it a name, e.g. `mscocoai-parquets`.

**Session 2+ (zero re-download):** add the promoted Dataset as an **input** to the
notebook, then:
```
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.train \
    --data-dir /kaggle/input/mscocoai-parquets --data-layout parquet \
    --task multitask --backbone resnet50 --out-dir outputs/run1 --resume auto
```
No extraction, no 7.5 GB again — read straight from the parquet shards.

If you are fine with the plain HF download each session instead, just pass no
`--data-dir` and set `HF_DATASETS_CACHE=/kaggle/working/hf_cache` (kept in
`/kaggle/working` so it persists across sessions).

## 2. Google Drive auto-save (optional, no admin)

Two routes, pick one:

**A. Kaggle Save Version** — the simplest. Click **Save Version** when you stop;
`/kaggle/working/outputs/...` is kept.

**B. Apps Script uploader** — per-epoch backup, survives a lost/timed-out session:
1. Follow `apps_script/uploader.gs` (5 minutes, runs as your Google account; no
   Cloud project, no OAuth client, no service account).
2. Add two **Kaggle Secrets** (notebook → Add-ons → Secrets):
   `MSCOCOAI_APPS_SCRIPT_URL` and `MSCOCOAI_APPS_SCRIPT_TOKEN`.
3. Add a cell before training:
   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   s = UserSecretsClient()
   os.environ["MSCOCOAI_APPS_SCRIPT_URL"] = s.get_secret("MSCOCOAI_APPS_SCRIPT_URL")
   os.environ["MSCOCOAI_APPS_SCRIPT_TOKEN"] = s.get_secret("MSCOCOAI_APPS_SCRIPT_TOKEN")
   ```
4. Train with `--drive-backend apps_script --upload-every epoch`.

Secrets are never printed, never written to disk, never committed. The upload
issues print only a warning and training continues if Drive is unreachable.

## 3. Train

Paper protocol (see `scripts/run_paper_models.sh` for all runs):

```
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.train \
    --task binary --backbone resnet50 --out-dir outputs/resnet50_binary \
    --resume auto --balance-loss --max-hours 11 \
    --drive-backend apps_script --upload-every epoch --save-steps 500
```

- `--max-hours 11` stops **cleanly** before the session limit (checkpoint +
  Drive upload + `results.json` written, then exits 0).
- `--balance-loss` uses pos_weight for the 1:5 real:fake skew in Label_A.
- `--save-steps 500` writes a checkpoint mid-epoch (T4 epochs are long).
- OOM? `--batch-size 16 --accum-steps 4` (same effective batch 64).
- ViT-L/14 (CLIP): use `--batch-size 16 --accum-steps 4`.
- Results stream to `outputs/<run>/history.jsonl`; a human-readable table is in
  `outputs/<run>/summary.md`.
- Uploading to Drive also ships every epoch's `history.jsonl`, `metrics.csv` and
  `results.json` automatically (disable with `--no-upload-metrics`).

## 4. What persists where

- `/kaggle/working/` — saved only on **Save Version** (or the committed run).
  Unsaved sessions are lost on timeout/disconnect.
- `--drive-backend apps_script` pushes fp16 `last_weights.pt` + result files to
  Drive every epoch; fully automatic.
- `--data-dir /kaggle/input/<dataset>` — persists as a Kaggle Dataset you attach
  as input; zero cost per session.
- Never commit outputs or secrets to the git repo.

## 5. Resume after a timeout (fresh session)

```
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.drive_upload \
    --action download-latest --run resnet50_binary \
    --out outputs/resnet50_binary/last_weights.pt
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.train \
    --task binary --backbone resnet50 --out-dir outputs/resnet50_binary \
    --resume weights-only ...
```

The restored run continues from the last intact epoch (its history.jsonl resumes
cleanly — no duplicated or half-written rows).

If your outputs survived in `/kaggle/working`, just rerun with `--resume auto`.
`--resume weights-only` restores weights only (fresh optimizer/scheduler).

## 6. Evaluate (final; test read exactly once)

```
!cd /kaggle/working/DeepFakeImageDetection && python -m mscocoai.evaluate \
    --checkpoint outputs/resnet50_binary/best.pt --split test \
    --report-dir reports/resnet50_binary_test
```

`report.json`, `per_generator.csv`, `confusion_matrix.png`, misclassified grid —
in `reports/`. Then **Save Version**.

## 7. CPU-only sanity run (never a paper run)

Useful only to check the pipeline; a ResNet-50 epoch on 42k images is ~1–1.5 h on
the 4-vCPU CPU tier. If you must:

```
!python -m mscocoai.train --task binary --backbone resnet18 --img-size 64 \
    --data-dir /kaggle/input/mscocoai-parquets --data-layout parquet \
    --epochs 1 --limit 200 --device cpu --batch-size 8 --num-workers 2 \
    --out-dir outputs/cpu_smoke
```

## 8. Multi-GPU (T4 ×2)

```
!NPROC=2 bash /kaggle/working/DeepFakeImageDetection/scripts/train_ddp.sh \
    --task binary --backbone resnet50 --data-dir /kaggle/input/mscocoai-parquets \
    --data-layout parquet --out-dir outputs/resnet50_binary_ddp --resume auto
```

## Honest notes

- Free-tier verification is possible only end-to-end on your own sessions; this
  guide documents the flow that was tested on the lab server, not a paid Kaggle
  guarantee.
- TPU: `--device tpu` exists (TPU v3-8) via `scripts/train_tpu.sh` but the TPU is
  **untested on real TPU hardware** — treat it as experimental.
- Do not redistribute MS-COCOAI images; the uploader only moves checkpoints/logs.
- If your Workspace blocks "Anyone" access in Apps Script, use **Anyone with Google
  account**; otherwise rely on Save Version plus the Kaggle-Dataset route.
