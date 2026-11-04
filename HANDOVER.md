# Handover — one page

**Repo:** https://github.com/AnnusShahzad/DeepFakeImageDetection.git
**Task:** train MS-COCOAI models (binary detection + 6-class attribution) and try to
beat the published 0.80 / 0.45 baselines; produce the numbers, not claims of them.

**Quick start per platform**

| You use | Read | Then run |
|---|---|---|
| Kaggle (free) | `kaggle/KAGGLE_GUIDE.md` | clone → `!pip install -q -r requirements.txt` → `python -m mscocoai.download --workers 8` → `python -m mscocoai.train ... --max-hours 11` |
| Google Colab (free) | `colab/COLAB_GUIDE.md` | upload `colab/mscocoai_colab.ipynb` → run top to bottom |
| Shared A4000 PC | `lab_pc/A4000_SETUP.md` | SSH → venv → download → `python -m mscocoai.launcher --jobs jobs.json --max-concurrency 1` |
| Anything else | `TEAM_GUIDE.md` | `python scripts/quickstart.sh` |

**Must set** (pick one): Kaggle Save Version, or Apps Script
(`apps_script/uploader.gs`, 5 min, runs as you, no admin — or ask the maintainer
for the URL/token), or Drive-for-Desktop `copy` backend. **Each epoch uploads the
fp16 weights and `results.json` automatically.**

**Must return** per run:
1. `outputs/<run>/results.json`
2. `outputs/<run>/summary.md`
3. `outputs/<run>/report/` (complete `evaluate` output, run **once** at the end)

**Parallel runs:** build a `jobs.json` with one entry per experiment →
`python -m mscocoai.launcher --jobs jobs.json` (VRAM is estimated per job;
finished jobs are skipped on restart).

**Ablation grid** (for the paper): `bash scripts/run_ablation.sh --scaled`
first (fast check), then without `--scaled`. Aggregate:
`python scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md`.

**Rules**

- Never commit `outputs/`, checkpoints, images, or secrets (they are gitignored).
- Test split read **once**, at the very end. Everything else is validation.
- No secrets in git: Apps Script token lives in your shell env / Kaggle Secrets.
- Label_A is 1:5 → read the **balanced** metric in the report, not raw accuracy
  (an always-predict-fake model scores 83% on the raw split).

**Status of verifications**

- 102 unit tests pass on the dev box.
- Lab A4000 run log is in `lab_pc/RUN_LOG.md` once the provisioning finishes.
- TPU path: written, **untested on real TPU hardware** (marked everywhere).
- Kaggle Dataset attachment and Colab runtime: user actions — treated as
  unverified in the guides.
