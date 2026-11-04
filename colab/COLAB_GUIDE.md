# Running MS-COCOAI on Google Colab (free tier)

Colab is usable for the **training pipeline** — not for reading the benchmark in
any interactive sense, and the compute is small (a free T4). Everything is
copy-paste; **use `colab/mscocoai_colab.ipynb` for a ready-made notebook.**

## 0. What the free Colab tier gives you

| Resource | Reality |
|---|---|
| GPU | Tesla T4 (≈15 GB VRAM) |
| RAM | ≈12–13 GB |
| Disk | `/content` is ephemeral (wiped on runtime reset); `/content/drive` persists |
| Time | runtimes disconnect when idle (~90 min) or after hours; **checkpoints are mandatory** |

## 1. Set up

Open `colab/mscocoai_colab.ipynb` (File → Upload notebook) and run the cells top
to bottom:

1. **Install:** `!pip install -q -r requirements.txt` from the repo (git clone or
   upload the repo as a Drive zip and unzip into `/content`).
2. **Mount Drive:** `from google.colab import drive; drive.mount('/content/drive')`
3. **Dataset:** keep the **17 parquet shards** in your Drive (few large files —
   Drive handles those fine) and pass `--data-dir /content/drive/MyDrive/mscocoai --data-layout parquet`.
   Do **not** export 96,000 small JPEGs into Drive; Drive-for-Desktop sync and
   Colab's FUSE layer both crawl under that many files.

## 2. Train

```python
!cd /content/mscocoai && python -m mscocoai.train \
    --task multitask --backbone convnext_tiny --freq-branch dct \
    --data-dir /content/drive/MyDrive/mscocoai --data-layout parquet \
    --num-workers 8 --batch-size 32 --accum-steps 2 \
    --max-hours 8 --ema-decay 0.999 --label-smoothing 0.05 \
    --drive-backend copy --drive-dir /content/drive/MyDrive/mscocoai/outputs \
    --out-dir /content/drive/MyDrive/mscocoai/outputs/mt_freq \
    --resume auto
```

- `--max-hours 8` stops the run **cleanly** (checkpoint + Drive upload + summary)
  before Colab kills the runtime.
- `--drive-backend copy --drive-dir /content/drive/MyDrive/mscocoai/outputs`
  copies each epoch's `last_weights.pt` + result files into Drive automatically —
  no keys, no Apps Script, just your own Drive.
- If the runtime disconnects, re-open the notebook, run the *install* cell again,
  and re-run the *train* cell with `--resume auto` — it continues from the last
  epoch.

## 3. Honest notes

- Colab gives no clean number for "how long a session lasts". Budget conservatively
  and checkpoint often (`--save-steps 500`).
- Free Colab's bandwidth to Drive is decent for the parquet files, but every
  epoch re-reads images if you train from Drive (FUSE); it works, it is just slower
  than a local mirror.
- The full ablation grid is too much for a free Colab week; use Colab for one or
  two runs, and the Kaggle/lab-PC flow for the grid.
- **No secrets in code cells** — keep your Drive folder private; nothing else is
  needed for the copy backend.
