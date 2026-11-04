#!/usr/bin/env bash
# One-command quickstart for MS-COCOAI — detects the platform and prints the
# exact next commands. No secrets required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d /kaggle/working ]]; then
    PLATFORM="kaggle"
elif [[ -d /content ]]; then
    PLATFORM="colab"
else
    PLATFORM="local"
fi

cat <<EOF
==============================================================
 MS-COCOAI quickstart
==============================================================
Platform: ${PLATFORM}
Repository: ${REPO_ROOT}

Next steps:

1. Install dependencies
     python3 -m pip install -r requirements.txt
   (Colab/Kaggle notebooks: \`!pip install -q -r requirements.txt\`)

2. Preflight check
     python3 -m mscocoai.doctor

3. Sanity-check the label mapping (no download needed)
     python3 scripts/print_labels.py

4. Download the dataset (parallel, resumable)
     python3 -m mscocoai.download --workers 8 --export-images ./data_mirror

5. Smoke-test a run (tiny, fast)
     python3 -m mscocoai.train --task multitask --backbone resnet18 \\
         --data-dir ./data_mirror --img-size 64 --epochs 2 --limit 500 \\
         --batch-size 16 --num-workers 4 --out-dir outputs/smoke

6. Resume after an interruption
     python3 -m mscocoai.train ... --resume auto
EOF
echo
if [[ "${PLATFORM}" == "kaggle" ]]; then
    cat <<'EOF'
Kaggle notes:
  - Enable Internet (required for the HF download).
  - Session 1 downloads the dataset; then Save Version -> push to a Kaggle Dataset.
  - Later sessions attach the Dataset and pass --data-dir /kaggle/input/<name> --data-layout parquet.
  - Use --max-hours 11 to stop cleanly before the 12 h session limit.
EOF
elif [[ "${PLATFORM}" == "colab" ]]; then
    cat <<'EOF'
Colab notes:
  - Mount Drive: from google.colab import drive; drive.mount('/content/drive')
  - Keep the parquet shards in Drive and use --data-layout parquet (Drive is slow for many small files).
  - Use --max-hours <hours> to checkpoint before a runtime reset.
EOF
fi
