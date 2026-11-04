# MS-COCOAI — AI-Generated Image Detection & Generator Attribution

Code for the experiments in `OBJECTIVE.md` (paper: *AI-Generated Image Detection
and Generator Attribution: A Unified Study on MS-COCO-Derived Synthetic Images*).
Runnable unchanged on **Kaggle**, **Google Colab** and a **shared lab PC** — with
**automatic checkpoint + result backup to Google Drive that needs no admin
access** (no Google Cloud project, no OAuth client, no service account), and a
**parallel, resumable dataset download**.

Dataset: [`Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
— 96,000 images, official splits 42,000 train / 9,000 validation / 45,000 test,
≈7.5 GB parquet, labels `Label_A` (0 real / 1 AI-generated) and `Label_B`
(0 real, 1 SD2.1, 2 SDXL, 3 SD3, 4 DALL-E 3, 5 MJ6).

**Baselines to beat (dataset paper): 0.80 binary detection ACC · 0.45 attribution ACC.**
Attribution (0.45) is directly comparable — Label_B is balanced at 16k per class.
Detection (0.80) is **not** comparable on the raw test split: Label_A is 1:5
(real:fake), so an always-predict-fake classifier already scores **83.3%** on it.
All detection results in this repo are therefore reported on the **balanced-subset
protocol** (7,500 real + 7,500 fake, seeded) with raw-skew metrics beside — see
*Metric reporting protocol* below.

## What this repo gives you

| Need | Where |
|---|---|
| Crash-proof, resumable training (history, metrics, checkpoints) | `mscocoai/results.py`, `mscocoai/engine.py` |
| Parallel resumable dataset download | `python -m mscocoai.download` |
| Kaggle **free tier** accepted path | `kaggle/KAGGLE_GUIDE.md` |
| Google Colab **free tier** accepted path | `colab/COLAB_GUIDE.md`, `colab/mscocoai_colab.ipynb` |
| Shared-PC / lab A4000 parallel jobs | `python -m mscocoai.launcher`, `lab_pc/A4000_SETUP.md` |
| Multi-GPU training (DDP) | `bash scripts/train_ddp.sh` |
| TPU (experimental, untested) | `--device tpu`, `bash scripts/train_tpu.sh` |
| SOTA levers, composable | `--freq-branch`, `--clip-init`, `--tta`, `--calibrate`, `--ema-decay`, `--label-smoothing` |
| No-admin Drive backup | `apps_script/uploader.gs`, `mscocoai/drive_upload.py` |
| Hand-over file for your team | `TEAM_GUIDE.md`, `HANDOVER.md` |

## Quickstart

```bash
python3 -m venv ~/venvs/mscocoai && source ~/venvs/mscocoai/bin/activate
pip install -r requirements.txt

python -m mscocoai.doctor                     # preflight: GPU/RAM/disk/internet — fails loudly if blocked
python scripts/print_labels.py                # label-map sanity, no download
python -m mscocoai.download --workers 8 --export-images ./data_mirror   # resumable download
python -m mscocoai.train --task multitask --backbone resnet18 --img-size 64 \
    --data-dir ./data_mirror --limit 500 --epochs 1 --out-dir outputs/smoke
python -m mscocoai.train ... --resume auto       # continues after any interruption
```

Track the work on Kaggle/Colab/the lab PC: see the three guides —
`kaggle/KAGGLE_GUIDE.md`, `colab/COLAB_GUIDE.md`, `lab_pc/A4000_SETUP.md` —
or `TEAM_GUIDE.md` for the hand-over one-pager.

## Repository layout

| Path | Purpose |
|---|---|
| `mscocoai/data.py` | HF hub / folder mirror / parquet-directory sources (one sample contract), paper augmentations, LOGO holdout filtering |
| `mscocoai/download.py` | **parallel resumable** snapshot download + verbatim-bytes mirror export (`manifest.json` resume) |
| `mscocoai/doctor.py` | environment preflight (fails loudly on blockers) |
| `mscocoai/engine.py` | shared training loop: AMP, grad-accum, AdamW + cosine, early stop, crash-proof resume, DDP, time budget |
| `mscocoai/models.py` | timm factory, `MultiTaskNet` (optional **frequency branch**), `ClipTaskModel` (CLIP ViT-B/16 / L-14), self-describing checkpoints |
| `mscocoai/frequency.py` | DCT-II branch / high-pass branch, de-normalises to true 0..1 pixels |
| `mscocoai/results.py` | keyed, crash-tolerant result log: `history.jsonl`, `metrics.csv`, `results.json`, `summary.md` |
| `mscocoai/distributed.py` | DDP init / rank-0-only writes / cross-rank metrics |
| `mscocoai/xla.py` | TPU (torch_xla) device path — **untested on real TPU hardware** |
| `mscocoai/tta.py` | test-time augmentation (identity / hflip / flip+scale) |
| `mscocoai/calibrate.py` | validation-fitted binary decision threshold (never test) |
| `mscocoai/launcher.py` | parallel job queue for a shared PC (per-job dirs/logs, VRAM estimate) |
| `mscocoai/metrics.py` | ACC/P/R/F1/AUC/PR-AUC/FPR@95TPR, balanced-subset report |
| `mscocoai/evaluate.py` · `report.py` | reports, per-generator detection, robustness, misclassified grid |
| `mscocoai/train.py` | the CLI |
| `scripts/run_paper_models.sh` | paper runs unchanged |
| `scripts/run_ablation.sh` | the ablation grid (`--scaled` quick, default full) |
| `scripts/collect_results.py` | aggregate `results.json` from many runs |
| `tests/` | 102 tests incl. crash-resume and parquet round-trip; run with `pytest` |

## Paper ↔ code map

| Paper element | Implementation |
|---|---|
| ResNet-50 / EfficientNet-B0, ImageNet init, AdamW, cosine, ≤30 epochs, patience 5, eff. batch 64 | `mscocoai/train.py` + `scripts/run_paper_models.sh` |
| Frozen CLIP-B/16 linear probe (O2) | `python -m mscocoai.clip_probe` |
| Frequency-aware dual-branch multi-task (the added component) | `--task multitask --freq-branch dct` |
| CLIP ViT-L/14 fine-tuned branch | `--clip-init clip_vitl14 --clip-unfreeze last2` |
| Multi-task heads λd = λa = 1 | `--lambda-d 1.0 --lambda-a 1.0` |
| Official splits; test read exactly once | validation-only selection; `evaluate.py` is the only test-reader |
| Robustness eval (JPEG-75 / blur) | `--degradation jpeg75\|blur` on evaluation |
| Per-generator difficulty (O5) | `mscocoai/evaluate.py` → `per_generator.csv`, confusion matrices, misclassified grid |
| LOGO generalisation protocol | `--holdout-generator sd3` (trains/val exclude that generator) |

## Metric reporting protocol

For binary detection every report carries **both**:

- **raw**: accuracy on the natural 1:5 test split (always reported beside,
  never alone),
- **balanced**: accuracy on a class-balanced subset (7,500 real + 7,500 fake,
  seeded) — directly comparable with the dataset paper's 0.80 baseline.

Attribution is reported as-is (Label_B is balanced). All final numbers land in
`results/` and in this README — filled from real runs, never projected.

## Results

*All rows below are **pending a real run** — this repo ships the pipeline, the
protocol and the commands; nobody has run the full grid yet.*

| Configuration | detection ACC | attribution ACC | notes |
|---|---|---|---|
| ResNet-50 (paper protocol) | pending | pending | O1/O3 reproduction |
| Multi-task ConvNeXt-Tiny + freq branch | pending | pending | the headline configuration |
| + EMA + label smoothing | pending | pending | schedule ablation |
| + TTA + calibrated threshold | pending | pending | inference ablation |
| + CLIP ViT-L branch | pending | pending | foundation-model ablation |

Run the grid (`bash scripts/run_ablation.sh --scaled` first on a single GPU,
then without `--scaled` in the background), aggregate with
`python scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md`.

## Verification

```bash
python -m pytest                     # 102 tests, CPU, no network
bash scripts/run_paper_models.sh     # paper reproduction
python -m mscocoai.doctor            # preflight
```

Crash/resume testing covers: truncated final `history.jsonl` lines, mid-epoch
checkpointing, `results.json` regeneration, and the Drive-upload dry-run.

## Honest notes

- **"SOTA" is claimed only with a run behind it.** Until `results/` has real
  numbers, this README says *pending*, not *SOTA*.
- The TPU path is written and importable but **untested on real TPU hardware**.
- Kaggle/Colab flows are documented from the lab-server verification run; the
  Kaggle-Dataset attachment step and Colab runtime are user actions that cannot
  be exercised from this repo (marked **unverified** on those steps).
- Apps Script caps requests at ~50 MB; parts are ≤24 MB with client-side
  reassembly. Kaggle persistence only works on **Save Version** — Drive upload
  during training is the real cross-session safety net.
- `../docs` outside this project are the user's files; untouched by this work.
- MS-COCOAI is research-use only; don't redistribute images.

## License & use

Code: MIT-0 / do what you want, no warranty.
Dataset license: see the MS-COCOAI dataset page (research use; contact the
authors for redistribution rights).
