# MS-COCOAI: Kaggle/A4000 Training Pipeline — Design

Date: 2026-11-04
Owner: Muhammad Annus Shahzad (with project "DLP — OBJECTIVE.md")

## Intent

Deliver a runnable, reproducible training and evaluation codebase for the MS-COCOAI
benchmark (the Defactify_Image_Dataset on Hugging Face; 96,000 images; splits
42k/9k/45k; labels `Label_A` binary, `Label_B` 6-class) that a university team can
run on Kaggle notebooks and on a shared lab PC with a single A4000, reproduce the
paper's four models, and push past the published baseline (0.80 detection ACC,
0.45 attribution ACC) toward SOTA on both tasks — without any Google Cloud project,
Cloud API key, or Workspace admin.

## Goals

- G1: All paper experiments as one-command runs: ResNet-50 (binary + 6-class),
  EfficientNet-B0 (binary + 6-class), frozen CLIP ViT-B/16 + linear probe, and a
  shared-backbone multi-task EfficientNet-B0 (λd = λa = 1).
- G2: Identical official splits and protocol for every model (224×224, ImageNet
  or CLIP normalization, deterministic val/test transforms, test untouched until
  the end).
- G3: SOTA levers as explicit flags, off for paper-faithful runs: EMA weights,
  label smoothing, any timm backbone (e.g. efficientnet_b4, convnextv2_base),
  CLIP `--clip-train last2|full` fine-tuning, mixup/cutmix, and a logit-ensemble
  evaluator.
- G4: Checkpoint auto-save to Google Drive with zero admin, three backends:
  (a) Kaggle Save Version output, (b) copy into a Google Drive for Desktop synced
  folder (lab PC), (c) Apps Script web-app uploader (chunked, works from Kaggle,
  needs only the user's university Google account — no Cloud project, no API key,
  no OAuth client).
- G5: Full evaluation report per checkpoint: ACC/F1/AUC/precision/recall
  (binary), ACC/macro-F1/confusion matrix (attribution), per-generator detection
  metrics, robustness pass (JPEG q75, Gaussian blur), qualitative
  success/failure montage, JSON + PNG artifacts.
- G6: Resumable training (optimizer/scheduler/scaler/EMA/best metric), so Kaggle
  session limits do not lose progress. Autosave runs *during* training: every
  epoch and every `--save-steps` optimizer steps, atomically write `last.pt`
  (full resume state) and a small fp16 weights-only `last_weights.pt`, and push
  it to Google Drive via the selected backend (`copy` on the lab PC; Apps Script
  chunked upload on Kaggle). A fresh Kaggle session restores with
  `python -m mscocoai.drive_upload --action download-latest --run <name>` and
  `--resume auto` picks up from `last.pt`; if only weights are available, training
  restarts from the weights with a fresh optimizer (flag `--resume-weights-only`).
  Note: /kaggle/working does not survive a lost session, so in-training Drive
  upload is the actual cross-session safety net — not a post-training step.
- G6a: Robustness to unexpected shutdown/power loss: all checkpoint writes are
  atomic (temp file + os.replace, fsync before rename), a `LATEST` marker file
  points at the last good checkpoint, and corrupt-file detection on resume
  (size + torch.load try/except) falls back to the previous checkpoint.

## Non-goals / anti-goals

- No claim of "SOTA" before real runs; code enables it, results decide it.
- Not re-splitting or re-labelling MS-COCOAI; captions are not used as model input.
- No Google Cloud service account, GCS bucket, or OAuth client — explicitly
  excluded per user constraints.
- No hosted service / serving stack; training and evaluation only.

## Constraints

- Dataset: public HF id `Rajarshi-Roy-research/Defactify_Image_Dataset`
  (~7.5 GB parquet, requires internet on Kaggle / lab PC). Label_B map:
  0 real, 1 SD2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 MJ6. Splits: 42k/9k/45k.
  Binary Label_A is 1:5 (real:fake) — reported as-is; optional balanced
  sampling/loss via flag.
- Kaggle: 9–12 h GPU sessions, ~57 GB disk in /kaggle/working, internet must be
  toggled on, T4/P100-class GPUs. Lab PC: 1× A4000 (16 GB), likely internet.
- University Google account: Apps Script + Drive only; no GCP admin.
- No secrets committed to git; uploader token via env var / Kaggle Secrets.

## Approaches considered

1. **Single monolithic notebook.** Fastest to paste on Kaggle, terrible to test
   and reuse on the A4000; every fix needs re-pasting. Rejected.
2. **Pip-installable package with module entry points.** Testable here on CPU
   (tiny synthetic dataset smoke tests), importable everywhere, same code on
   Kaggle and lab PC. Chosen.
3. **Apps Script for Drive upload (chosen) vs rclone vs service account.**
   rclone's shared client_id is retired during 2026; own OAuth client requires a
   Cloud project — both ruled out. Service accounts need GCP admin — ruled out.
   Apps Script runs as the user's own account; a ~60-line `uploader.gs` plus a
   ~150-line `mscocoai/drive_upload.py` gives chunked upload of arbitrarily large
   checkpoints (32 MB parts, base64, `DriveApp.createFile`), a finalize step that
   concatenates parts inside the script when small, or leaves ordered part files
   plus a printed `cat` instruction for large ones (fp16 export keeps everything
   small except ResNet-50). Kaggle Save Version and Drive-for-Desktop copy are
   kept as simpler defaults.

## What gets built (structure)

```
OBJECTIVE.md                    (unchanged)
README.md                       (what this is, paper↔file map, results table)
requirements.txt                (torch, timm, open_clip_torch, datasets, …)
mscocoai/
  data.py        (HF or local-folder loader, paper transforms + JPEG/blur aug,
                  CLIP/ImageNet norm, pos_weight, balanced sampler, workers)
  models.py      (timm CNN factory, multi-task EfficientNet-B0, CLIP probe head,
                  label maps, checkpoint export: fp32/fp16 + optional EMA)
  engine.py      (AMP train loop, grad accum ×2, AdamW, 1-epoch warmup + cosine,
                  early stop (val loss, patience 5/3), EMA, resume, history.jsonl)
  metrics.py     (ACC/P/R/F1/AUC/PR-AUC/FPR@95TPR, macro-F1, 6×6 confusion,
                  per-generator detection)
  train.py       (python -m mscocoai.train --backbone resnet50 --task binary …)
  evaluate.py    (python -m mscocoai.evaluate --task … --split test --report dir)
  clip_probe.py  (python -m mscocoai.clip_probe — extract embeddings to .npy,
                  cache, train linear probe, identical protocol)
  ensemble.py    (python -m mscocoai.ensemble — logit/prob averaging, full report)
  drive_upload.py(upload/download with --action upload|download|download-latest,
                  --backend apps_script|copy|kaggle; --dry-run; 32 MB base64
                  chunks; parts concatenated client-side on download)
  report.py      (shared report writers: JSON, confusion PNG, montage grid)
configs/                        (one .yaml per paper model + sota_ examples)
apps_script/uploader.gs         (paste-into-script.google.com, step-by-step)
kaggle/KAGGLE_GUIDE.md          (exact notebook commands, secrets, Save Version,
                                  resume, internet toggle, disk notes)
lab_pc/DRIVE_FOR_DESKTOP.md     (point --drive-dir at synced folder; auto-sync)
scripts/print_labels.py         (sanity-check label mappings/split sizes)
tests/test_smoke.py             (CPU, tiny synthetic dataset, end-to-end)
docs/superpowers/specs/…-design.md (this file)
```

Kaggle usage: upload/clone the folder, `pip install -r requirements.txt`,
`HF_HUB_ENABLE_HF_TRANSFER=1 python -m mscocoai.train …`, weights and reports in
`/kaggle/working/outputs/`; `python -m mscocoai.drive_upload --backend apps_script`
or rely on Save Version. Lab PC: same, plus `--drive-dir "/mnt/g/My Drive/DLP/…"`.

## Verification

- `python -m py_compile` all modules; `pytest tests/test_smoke.py` end-to-end on a
  tiny in-repo synthetic "dataset" (no downloads), proving data→train→eval→upload
  dry-run works, plus an interrupt/resume cycle (train 1 epoch, kill, resume with
  `--resume auto`, verify it continues from the saved epoch/steps).
- README includes the exact commands for (1) preflight checks, (2) downloading the
  dataset (and offline/local-folder fallback), (3) all six training runs, (4) final
  test-only evaluation, (5) Drive upload (all three backends), (6) resume after a
  Kaggle timeout.
- Report artifacts are produced by the same code path in the smoke test and on the
  real runs.

## Risks / open points

- Kaggle disk (57 GB) vs 7.5 GB dataset + dataset cache copy + checkpoints —
  mitigated by `HF_DATASETS_CACHE=/kaggle/temp/hf` and fp16 exports.
- Apps Script university policy may block "Anyone" access; fallback is
  "Anyone with Google account" or the copy/Kaggle backends. Documented.
- MS-COCOAI license: research use only; no redistribution of images.
