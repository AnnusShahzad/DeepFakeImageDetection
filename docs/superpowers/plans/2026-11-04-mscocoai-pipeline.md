# MS-COCOAI Training Pipeline Implementation Plan

> **For agentic workers:** Native implementation in this session is expected (user-approved design). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the MS-COCOAI (Defactify_Image_Dataset) training + evaluation codebase for Kaggle and the lab A4000 PC, faithful to the paper plus SOTA levers, with admin-free Google Drive checkpoint autosave and resume.

**Architecture:** One importable package `mscocoai/` with module entry points (`python -m mscocoai.train …`). Data streams from Hugging Face (`Rajarshi-Roy-research/Defactify_Image_Dataset`) or a local folder mirror. All models share one training engine (AMP, cosine+warmup, early stop, atomic checkpoints, EMA, optional Drive upload hook). Evaluation is a separate offline entry point that never touches the test split during training selection.

**Tech Stack:** Python 3.10+, PyTorch + torchvision, timm, open_clip_torch, datasets (HF), scikit-learn, numpy/pandas/matplotlib, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-mscocoai-training-pipeline-design.md`

## Global Constraints

- Dataset id exactly `Rajarshi-Roy-research/Defactify_Image_Dataset`; splits train=42000, validation=9000, test=45000; never re-split, never evaluate test until final.
- Labels: `Label_A` 0=real, 1=AI-generated; `Label_B` 0=real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E 3, 5=MJ6.
- Input 224×224; CNN norm = ImageNet mean/std; CLIP norm = OpenAI mean/std; val/test transforms deterministic (Resize→CenterCrop); augmentation train-only.
- Batch 32 with grad accumulation ×2 (effective 64); AdamW lr=1e-4, wd=1e-4 (probe: lr=1e-3, wd=1e-4); 1-epoch linear warmup then cosine decay; ≤30 epochs; early stop on val loss, patience 5 (probe 3); mixed precision; seed 42.
- Paper models must run without any flag: `resnet50`, `efficientnet_b0`, frozen `clip_vitb16` + linear probe, `multitask_efficientnet_b0` (λd=λa=1).
- Checkpoints: atomic (tmp+fsync+`os.replace`), `best.pt`, `last.pt` (full resume incl. optimizer/scheduler/scaler/EMA/epoch), `last_weights.pt` (fp16), `LATEST` marker; resume must survive Kaggle session loss via Drive re-download.
- Google access limited to: Kaggle Save Version, Drive-for-Desktop copy, Apps Script uploader. No GCP/service account/OAuth client/rclone-shared-id (retired 2026).
- Images are not redistributed; uploaders move only checkpoints/reports.
- No network calls in tests; tests run on a synthetic local dataset.

## Review Focus

Pinned in the owning task:

1. **Wrong Label_B map** (SD3↔SDXL swaps) → Task 2 test `test_label_mapping`.
2. **Test-split leakage** during tuning (any test metric printed pre-final) → Task 4 engine has no test path; Task 6 `evaluate.py` is the only test reader; guide says so.
3. **Crash/partial checkpoint corrupts resume** → Task 4 tests `test_resume_after_corrupt_last_pt` (falls back to previous good `last.pt` via `LATEST`).
4. **Kaggle session loss wipes /kaggle/working** → Task 9 uploader runs in-training; Task 10 guide mandates `--upload-every epoch`; smoke test covers upload dry-run + restore.
5. **AUC/F1 computed on wrong positive class** (fake=1) or degraded by JPEG aug RNG → Task 5 test `test_binary_metrics_known_scores`, Task 2 `test_jpeg_aug_quality_range`.

---

### Task 1: Package scaffold + constants + synthetic fixture
**Files:** Create `mscocoai/__init__.py`, `mscocoai/constants.py`, `requirements.txt`, `.gitignore`, `tests/conftest.py` (tiny synthetic dataset factory: 6 label dirs × 8 imgs, 64×64 jpegs), `pyproject.toml` (pytest config).
- [ ] Step 1: Write failing test `tests/test_constants.py::test_label_names` asserting `LABEL_B_NAMES == ["real","sd21","sdxl","sd3","dalle3","mj6"]`, `NUM_ATTR_CLASSES==6`, `HF_DATASET_ID` exact.
- [ ] Step 2: Run `pytest tests/test_constants.py -v` → FAIL (module missing).
- [ ] Step 3: Implement `constants.py` with the above + `IMAGENET_MEAN/STD`, `CLIP_MEAN/STD`, `SPLIT_SIZES`.
- [ ] Step 4: Run test → PASS.
- [ ] Step 5: Commit: `git add . && git commit -m "chore: scaffold mscocoai package and constants"`

### Task 2: Data layer (HF + local folder, paper transforms)
**Files:** Create `mscocoai/data.py`; Test `tests/test_data.py`.
**Interfaces:** `build_dataset(split, *, data_dir=None, limit=None) -> Dataset`; `build_loader(...)->DataLoader`; `get_transforms(kind, img_size=224)`; `compute_pos_weight(labels)->float`; classes `HfImageDataset`, `LocalFolderDataset`.
- [ ] Step 1: Failing tests: `test_label_mapping` (HF sample dict → (label_a,label_b) ints; local folder names map to 0..5); `test_jpeg_aug_quality_range` (encode/decode returns RGB Image, quality stays in [50,95]); `test_local_folder_split_sizes`; `test_pos_weight_five_to_one` (expect ≈0.2 for 1:5).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement `data.py`: local loader layout `root/{train,validation,test}/{real,sd21,sdxl,sd3,dalle3,mj6}/*.{jpg,png,webp}`; HF loader via `datasets.load_dataset(HF_DATASET_ID, split=split, cache_dir=env HF_DATASETS_CACHE)`; JPEG aug callable (quality randint 50–95 via BytesIO roundtrip); GaussianBlur p=0.25; RRC(scale=(0.8,1.0))+flip0.5+ColorJitter(0.2,0.2,0.2,0.02); val/test Resize(256)+CenterCrop(224); ImageNet/CLIP norm switches on `norm="imagenet"|"clip"`; `num_workers=4, pin_memory, persistent_workers`; `WeightedRandomSampler` for `--balance-sampler`.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: Commit `feat: data layer for HF and local mirrors`.

### Task 3: Models + atomic checkpoint I/O + EMA
**Files:** Create `mscocoai/models.py`; Test `tests/test_models.py`.
**Interfaces:** `build_backbone(name, num_classes, pretrained=True)` (timm); `MultiTaskEffB0(backbone="efficientnet_b0")` forward→`(z_d:[B], z_a:[B,6])`; `save_checkpoint(state, path)`; `load_checkpoint(path, map_location)`; `EMA(model, decay)` with `copy_to`/`store`/`restore`; `export_fp16(src, dst)`; `load_model_for_eval(ckpt_dir, task)`.
- [ ] Step 1: Failing tests: `test_cnn_forward_shape` (`build_backbone("efficientnet_b0",1)` on 2×3×224×224 → [2,1]); `test_multitask_shapes` (→ [2], [2,6]); `test_checkpoint_roundtrip` (save→load state equal, file not corrupt); `test_fp16_export` (dtype half, keys identical); `test_ema_updates_shadow`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement; all `torch.save` via temp file in same dir + fsync + `os.replace`; `load_checkpoint` verifies dict has `model` and raises `CheckpointCorrupt` with path.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: Commit `feat: model builders and crash-safe checkpointing`.

### Task 4: Engine (train/validate/resume/autosave)
**Files:** Create `mscocoai/engine.py`, `mscocoai/train.py`; Test `tests/test_engine.py`.
**Interfaces:** `fit(cfg) -> dict` (best val metrics); `train.py` CLI: `--task {binary,attribution,multitask} --backbone --epochs --out-dir --resume {auto,PATH,none} --save-steps --drive-backend {none,copy,apps_script} --drive-dir --upload-every {epoch,none}`.
- [ ] Step 1: Failing tests: `test_one_epoch_overfits_tiny` (16 imgs, loss decreases); `test_resume_continues_epoch` (run 1 epoch, save, resume, assert `start_epoch==1`); `test_resume_after_corrupt_last_pt` (truncate last.pt → `LATEST` points to previous-good or clean error + falls back); `test_early_stop_patience` (synthetic val loss plateau).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement: AMP (`torch.autocast('cuda')`, `GradScaler`), grad accum 2, AdamW, 1-epoch linear warmup + `CosineAnnealingLR`, early stop on val loss (patience arg, default 5), `history.jsonl`, checkpoint each epoch + every `--save-steps` steps, atomic everywhere, after every save optionally call `drive_upload.upload_file` (import-guarded, failures logged not fatal). `--resume auto` finds newest valid checkpoint in `--out-dir`; corrupt latest → try previous (keep `last.pt.bak`), never crash silently. Train NEVER reads the test split.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: Commit `feat: training engine with autosave and resume`.

### Task 5: Metrics + per-generator report
**Files:** Create `mscocoai/metrics.py`; Test `tests/test_metrics.py`.
**Interfaces:** `binary_report(y_true, y_score) -> dict` (acc, precision, recall, f1, auc, pr_auc, fpr_at_95tpr); `attribution_report(...) -> dict` (acc, macro_f1, per_class, confusion 6×6 list); `per_generator_detection(y_true, y_score, y_gen) -> dict`; `save_confusion_png(cm, path)`.
- [ ] Step 1: Failing tests: `test_binary_metrics_known_scores` (hand-computed acc/f1/auc=0.75 case, fpr@95 defined), `test_macro_f1_known`, `test_confusion_shape`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement with sklearn; macro-F1 = `f1_score(average="macro")`; FPR@95TPR from ROC curve (min FPR where TPR≥0.95); per-generator = binary metrics restricted to each fake generator + reals.
- [ ] Step 4: Run tests → PASS.
- [ ] Step 5: Commit `feat: evaluation metrics and per-generator breakdown`.

### Task 6: evaluate.py + report artifacts
**Files:** Create `mscocoai/evaluate.py`, `mscocoai/report.py`; Test `tests/test_evaluate.py`.
**Interfaces:** `python -m mscocoai.evaluate --task … --checkpoint … --split test --report-dir DIR [--robustness none,jpeg75,blur]` writes `report.json`, `confusion_matrix.png`, `per_generator.csv`, `samples_grid.png`.
- [ ] Step 1: Failing test: `test_eval_writes_artifacts` (train tiny model 1 epoch on synthetic data → run evaluate → assert the 4 files exist and JSON parses).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement: deterministic loader (task-appropriate), collects predictions in chunks under AMP, metrics via Task 5, 6×6 PNG (counts + row-normalized), montage of up to 24 misclassified images (true/pred labels in caption). `jpeg75`/`blur` variants apply the paper degradation before the val/test transform.
- [ ] Step 4: Run test → PASS.
- [ ] Step 5: Commit `feat: offline test evaluation and report artifacts`.

### Task 7: CLIP frozen probe (+ last2/full unfreeze)
**Files:** Create `mscocoai/clip_probe.py`; Test `tests/test_clip_probe.py`.
**Interfaces:** `python -m mscocoai.clip_probe --mode extract|train|eval --model ViT-B-16 --pretrained openai --out-dir DIR [--train-mode frozen|last2|full]`; embedding cache `<out-dir>/embeddings_{split}.npy`; probe = single Linear(768?→512 for B/16) trained with AdamW lr1e-3, wd1e-4, ≤15 epochs, patience 3 on val loss.
- [ ] Step 1: Failing test: `test_probe_trains_on_synthetic_embeddings` (random separable embeddings → probe reaches >0.9 val acc; cache roundtrip loads identical array).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement: `open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")`, eval() + freeze per `--train-mode`; extract with AMP in batches of 64; train loop mirrors engine settings (warmup+cosine, AMP optional since heads tiny); log to same metric/report stack (reuse Task 5). Test-time only via `evaluate.py` logic.
- [ ] Step 4: Run test → PASS.
- [ ] Step 5: Commit `feat: CLIP embedding extraction and linear probe`.

### Task 8: Ensemble evaluator
**Files:** Create `mscocoai/ensemble.py`; Test `tests/test_ensemble.py`.
**Interfaces:** `python -m mscocoai.ensemble --task binary|attribution --checkpoints a.pt b.pt … --split test --report-dir DIR`; averaging in probability space; weights optional via `--weights 2,1`.
- [ ] Step 1: Failing test: `test_ensemble_artifacts` (two tiny checkpoints → report.json written; ensemble AUC ≥ single-model AUC not asserted — artifacts + keys asserted only).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement: load each model via `load_model_for_eval`, run, average softmax/sigmoid probs, reuse Task 5/6 report writers.
- [ ] Step 4: Run test → PASS.
- [ ] Step 5: Commit `feat: multi-checkpoint logit ensemble evaluator`.

### Task 9: Drive upload/restore (Apps Script + copy) and in-training hook
**Files:** Create `mscocoai/drive_upload.py`, `apps_script/uploader.gs`; Test `tests/test_drive_upload.py`.
**Interfaces:** `python -m mscocoai.drive_upload --action upload|download|download-latest --file|--run --backend apps_script|copy|none [--drive-dir DIR] [--chunk-mb 32] [--dry-run]`; env `MSCOCOAI_APPS_SCRIPT_URL`, `MSCOCOAI_APPS_SCRIPT_TOKEN`; gs exposes `doPost` (upload parts, finalize merge if total ≤ limit else keep parts + concat manifest) and `doGet` (list/download parts base64).
- [ ] Step 1: Failing test: `test_chunk_plan_and_dry_run` (100 MB file → 4 chunks @32MB; dry-run prints plan, no network); `test_download_latest_manifest` (mock urlopen returning manifest JSON → correct filenames).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement client (urllib only, no new deps): base64 parts, JSON envelope `{token, action, run, name, index, total, data}`; retry ×3 with backoff; verify part count; download-latest pulls `LATEST` manifest then parts, concatenates in order, writes atomic; corrupt part → clear error. `gs`: secret token check, writes parts to folder `<FOLDER_NAME>/<run>/`, `doGet` returns base64 with 45 MB cap per response (auto parts). Also `copy` backend (atomic copy into `--drive-dir`). In-training hook: after each checkpoint save, if backend≠none and `--upload-every epoch`, call uploader for `last_weights.pt` + `last.pt`; failures warn only.
- [ ] Step 4: Run tests → PASS; `python -m mscocoai.drive_upload --action upload --file OBJECTIVE.md --backend copy --drive-dir /tmp/x` works.
- [ ] Step 5: Commit `feat: admin-free Google Drive checkpoint sync`.

### Task 10: Guides, scripts, README, final verification
**Files:** Create `README.md`, `kaggle/KAGGLE_GUIDE.md`, `lab_pc/A4000_SETUP.md`, `scripts/run_paper_models.sh`, `scripts/print_labels.py`.
(No YAML configs — paper commands live in `scripts/run_paper_models.sh` and README; single source of truth.)
- [x] Guides with copy-paste commands: Kaggle (internet on, HF cache in /kaggle/temp, resume after timeout via `download-latest` + `--resume weights-only`, Save Version); lab A4000 (HF cache on big disk, Drive for Desktop `--drive-backend copy`).
- [x] `python -m py_compile mscocoai/*.py` and `pytest -q` — 36/36 green.
- [x] README: paper↔file map, exact Label_B mapping, "how to beat the baseline" levers (EMA, last2 unfreeze, EffNet-B4/ConvNeXtV2, ensemble, balanced loss; note binary 1:5 skew), reports, honest caveats (license, no SOTA claim without runs).
- [x] End-to-end smoke of the actual CLIs: train → auto Drive copy → evaluate → download-restore (all on a tiny synthetic local dataset; tests cover resume-after-corruption).
- [x] No large files committed; checkpoints land in `outputs/` (git-ignored).
