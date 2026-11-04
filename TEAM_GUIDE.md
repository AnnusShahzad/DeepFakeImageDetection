# MS-COCOAI — Team guide (start here)

You have been handed a working training pipeline. This page is everything you
need. Ask the maintainer anything not covered here.

## What you can run

| Platform | How |
|---|---|
| **Kaggle (free)** | `kaggle/KAGGLE_GUIDE.md` — GPU, Internet On, Dataset attached as input |
| **Google Colab (free)** | `colab/COLAB_GUIDE.md` + `colab/mscocoai_colab.ipynb` |
| **Shared lab PC (A4000 16 GB)** | `lab_pc/A4000_SETUP.md` — SSH, one job per GPU, `mscocoai.launcher` for parallel jobs |
| **Your own machine** | `python scripts/quickstart.sh` — prints the exact next steps |

## The 3 things you must set

1. **Checkpoint backup** — pick ONE:
   - Kaggle Save Version (default, zero setup), or
   - Apps Script uploader (`apps_script/uploader.gs`, 5 min, no admin — best), or
   - Google Drive for Desktop + `--drive-backend copy` (lab PC).
   The code **never** sends images anywhere; only weights and result logs move.
2. **Run directory name** — one per experiment, e.g. `outputs/mt_freq`. Never
   *inside* the repo copy you plan to commit.
3. **Your slice** — one experiment per person; don't co-write a run directory.

## The 3 things you must return

1. `outputs/<your-run>/results.json` — the aggregate of everything the run measured.
2. `outputs/<your-run>/summary.md` — the human-readable table.
3. `outputs/<your-run>/report/` — `report.json`, `per_generator.csv`, confusion
   matrix PNGs (from the final `evaluate` run).

Zip those three and send them to the maintainer — or push a branch containing
**only** `results.json` (no `outputs/`, no images).

## The 4 commands you'll actually use

```bash
python -m mscocoai.doctor          # before anything: fails loudly if broken
python -m mscocoai.download --workers 8 --export-images ./data_mirror
python -m mscocoai.train --task multitask --backbone convnext_tiny \
    --freq-branch dct --data-dir ./data_mirror --out-dir outputs/<your-run>
python -m mscocoai.evaluate --checkpoint outputs/<your-run>/best.pt \
    --split test --report-dir reports/<your-run>-test   # only at the very end
```

## Parallel runs on the shared PC

```bash
python -m mscocoai.launcher --jobs jobs.json --max-concurrency 1
```

One full-model job at a time on 16 GB of VRAM. The launcher writes
`logs/<job>.log` and `jobs_status.json`; restart it any time — finished jobs are
skipped. CPU-only smoke jobs can share the machine.

## Hard rules

- **No images, no checkpoints, no secrets in git.** `.gitignore` already blocks
  them; if you add data, it is wrong.
- **No dataset redistribution.** The images stay in HF/Drive; only your metrics
  travel.
- **Test split** is read exactly once — for the final report only. Everything
  before that is validation.
- **Never** paste your Apps Script token or Drive credentials anywhere shared.
- If `doctor` says BLOCKED, fix the environment before touching the code.

## If something fails

1. Re-run `python -m mscocoai.doctor` and fix what it flags.
2. Check `outputs/<your-run>/history.jsonl` — every epoch is one line; an
   interruption mid-epoch leaves a `partial` row, which is normal.
3. `--resume auto` continues from the last intact epoch (checkpoints are atomic).
4. Logs: `logs/<job>.log` on the lab PC; on Kaggle, the cell output.
5. Anything reproducible but wrong → open an issue with the commands + output.

## What "done" means

- `results.json` shows `"stop_reason": "completed"` (or `"time_budget"` on Kaggle).
- `summary.md` lists your epochs.
- `report/` contains the final test evaluation.

Deliver those and you're done.
