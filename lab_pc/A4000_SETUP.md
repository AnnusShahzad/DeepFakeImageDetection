# Lab PC (shared, single A4000) — setup, Drive auto-save, parallel runs

This is the flow used on `fast@100.123.105.15` (host `LAB12-18`, RTX A4000 16 GB,
24 vCPU, 31 GB RAM, ≈110 GB free disk) and the same flow works on any shared
Linux PC with an NVIDIA GPU.

## 0. Environment

```bash
ssh -l fast 100.123.105.15
nvidia-smi                                             # confirm A4000 visible
python3 -m venv ~/venvs/mscocoai
~/venvs/mscocoai/bin/pip install --upgrade pip
# CUDA torch build (choose cu124/cu128 matching your driver):
~/venvs/mscocoai/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
cd ~/mscocoai
~/venvs/mscocoai/bin/pip install -r requirements.txt
~/venvs/mscocoai/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Per-commit note: the requirements install timm/open_clip/datasets/pyarrow; the
CPU-only pins in `requirements-lock.txt` document the exact verified set.

## 1. Sync the code

```bash
# from your laptop:
rsync -a --exclude=.git --exclude=outputs --exclude=data_mirror ./ fast@100.123.105.15:~/mscocoai/
ssh -l fast 100.123.105.15 'cd ~/mscocoai && ~/venvs/mscocoai/bin/python -m mscocoai.doctor'
```

`doctor` prints GPU/RAM/disk/internet/license checks and fails loudly if
anything blocks.

## 2. Dataset — parallel, resumable download

```bash
~/venvs/mscocoai/bin/python -m mscocoai.download --workers 8 \
    --cache-dir ~/hf_cache --export-images ~/mscocoai/data_mirror
```

- `--workers 8` (or 16 on a fast link) downloads the 17 parquet shards in
  parallel. The HF cache keeps `*.incomplete` partials so an interrupted download
  **resumes**, it never restarts from zero.
- `--export-images` writes the folder mirror
  (`~/mscocoai/data_mirror/<train|validation|test>/<class>/*.jpg`) with verbatim
  bytes and a `manifest.json` so the export itself is resumable. The dataloader
  then reads plain JPEGs from NVMe — much faster than parquet row decoding.
- `--verify` re-checks sizes against the repo metadata.
- On a **fast** link also set `HF_HUB_ENABLE_HF_TRANSFER=1` (done automatically
  when `hf_transfer` is installed) for chunk-parallel transfer inside each file.
- Expected: ~7.5 GB for the dataset + a bit more for the mirror; capacity exists
  on the server.

## 3. Google Drive auto-save (three options, best first)

**Option A — Google Drive for Desktop** (zero keys): install Drive for Desktop,
sign in, then train with
`--drive-backend copy --drive-dir "$HOME/Google Drive/My Drive/MscocoAI-Weights"`
— every epoch copies fp16 `last_weights.pt` + result logs into Drive.

**Option B — Apps Script** (no Drive-for-Desktop needed): 5-minute setup in
`apps_script/uploader.gs`, then `--drive-backend apps_script`. Set
`MSCOCOAI_APPS_SCRIPT_URL` and `MSCOCOAI_APPS_SCRIPT_TOKEN` in your shell only.

**Option C — plain disk**: just `outputs/` on the box; add `--drive-backend copy`
anyway — free insurance.

## 4. Recommended runs (A4000, 16 GB)

```bash
# smoke test first — everything must pass before real compute is burned
~/venvs/mscocoai/bin/python -m mscocoai.train --task multitask --backbone resnet18 \
  --data-dir ~/mscocoai/data_mirror --img-size 64 --epochs 2 --limit 2000 \
  --batch-size 32 --num-workers 8 --out-dir ~/mscocoai/outputs/smoke --resume auto

# frequency-aware dual-branch (the paper's headline component)
~/venvs/mscocoai/bin/python -m mscocoai.train --task multitask \
  --backbone convnext_tiny --freq-branch dct --freq-k 32 \
  --data-dir ~/mscocoai/data_mirror --num-workers 8 \
  --ema-decay 0.999 --label-smoothing 0.05 \
  --out-dir ~/mscocoai/outputs/mt_freq --resume auto \
  --drive-backend copy --drive-dir "$HOME/Google Drive/My Drive/MscocoAI-Weights"
```

Batch 32 @ 224 fits 16 GB comfortably. Bigger backbones (ViT-L/14): use
`--batch-size 16 --accum-steps 4`.

## 5. Shared PC — run jobs in parallel

Don't let two jobs fight over RAM and CPU; use the launcher (each job gets its own
`--out-dir`, log file and status slot; free VRAM is estimated to set concurrency):

```bash
# jobs.json — one entry per experiment
cat > jobs.json <<'JSON'
[
  {"name": "mt_freq",   "cmd": ["~/venvs/mscocoai/bin/python","-m","mscocoai.train","--task","multitask","--backbone","convnext_tiny","--freq-branch","dct","--data-dir","~/mscocoai/data_mirror","--out-dir","~/mscocoai/outputs/mt_freq","--resume","auto"], "out_dir": "~/mscocoai/outputs/mt_freq", "workers": 8},
  {"name": "paper_b50", "cmd": ["~/venvs/mscocoai/bin/python","-m","mscocoai.train","--task","binary","--backbone","resnet50","--data-dir","~/mscocoai/data_mirror","--out-dir","~/mscocoai/outputs/paper_b50","--resume","auto"], "out_dir": "~/mscocoai/outputs/paper_b50", "workers": 8}
]
JSON

~/venvs/mscocoai/bin/python -m mscocoai.launcher --jobs jobs.json --max-concurrency 1

# check on them without attaching
tail -f logs/mt_freq.log
cat jobs_status.json
```

- One full-model job at a time on a 16 GB A4000 (`--max-concurrency 1`); several
  CPU-only smoke jobs can run at once.
- Every job is launched in its own session (`start_new_session`) so it survives
  the SSH session; the launcher re-reads `jobs_status.json` and skips already
  finished jobs when restarted.
- Keep `--num-workers ≤ free_cores / jobs`.

## 6. Crash / power-cut / Ctrl-C recovery

```bash
# everything continues from the last intact epoch:
~/venvs/mscocoai/bin/python -m mscocoai.train ... --resume auto

# restore from Drive instead (weights-only; fresh optimizer):
~/venvs/mscocoai/bin/python -m mscocoai.drive_upload --action download-latest \
    --run mt_freq --out ~/mscocoai/outputs/mt_freq/last_weights.pt
~/venvs/mscocoai/bin/python -m mscocoai.train ... --resume weights-only
```

Saves are atomic (tmp → fsync → rename) plus a `LATEST` marker and `.bak`
fallback — a dead machine loses at most the current epoch.

## 7. After training — reports and aggregation

```bash
~/venvs/mscocoai/bin/python -m mscocoai.evaluate --checkpoint ~/mscocoai/outputs/mt_freq/best.pt \
    --split test --report-dir ~/mscocoai/reports/mt_freq_test
~/venvs/mscocoai/bin/python scripts/collect_results.py ~/mscocoai/outputs/* \
    --csv ~/mscocoai/results/ablation.csv --md ~/mscocoai/results/ablation.md
```

Read `results/ablation.md` to fill the README's results table.

## 8. JupyterLab front end

The same box serves JupyterLab at `http://100.123.105.15:8888`. Point the file
browser at `~/mscocoai`; the code, logs and reports are all there. Use it for
inspection, not for heavy imports (notebooks don't need the training loop).

## Honest notes

- One job per GPU at a time unless VRAM explicitly allows more; the shared PC is
  a shared PC.
- Disk needed ≈ 60 GB: 7.5 GB dataset + ~7.5 GB mirror + < 2 GB per run
  checkpoint set (fp16 exports keep Drive small).
- If the machine has no Drive for Desktop and you don't want Apps Script, rely on
  `outputs/` and copy manually — automatic is better; do it.
