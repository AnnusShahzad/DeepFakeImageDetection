#!/usr/bin/env bash
# The ablation grid for the paper: scaled (fast sanity) or full (publication numbers).
#
#   bash scripts/run_ablation.sh --scaled           # ~1 h on an A4000, 8 epochs, limit 5000
#   bash scripts/run_ablation.sh                    # full grid, 30 epochs, early stop 5
#
# Data: pass DATA_DIR (folder mirror or parquet dump) and MSCOCOAI_PYTHON.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCALED=0
KEEP_GOING=0
ONLY=""
DATA_DIR="${DATA_DIR:-data_mirror}"
PY="${MSCOCOAI_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scaled) SCALED=1; shift ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

LOGDIR=outputs/ablation
mkdir -p "$LOGDIR"

LIMIT_ARGS=()
if [[ "$SCALED" == "1" ]]; then
  LIMIT_ARGS=(--limit 5000 --epochs 8)
  echo "=== SCALED grid: 8 epochs, 5000 imgs/split ==="
else
  LIMIT_ARGS=(--epochs 30)
  echo "=== FULL grid: 30 epochs, early patience 5 ==="
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR not found: $DATA_DIR (run: python -m mscocoai.download --export-images $DATA_DIR)"
  exit 2
fi

run_row() {
  local name="$1"; shift
  if [[ -n "$ONLY" && ",$ONLY," != *",$name,"* ]]; then
    echo
    echo "=== ROW $name === skipped (--only $ONLY)"
    return 0
  fi
  echo
  echo "=== ROW $name ==="
  echo "$PY -m mscocoai.train --out-dir outputs/$name --data-dir $DATA_DIR --num-workers 8 ${LIMIT_ARGS[*]} $*"
  if ! "$PY" -m mscocoai.train --out-dir "outputs/$name" --data-dir "$DATA_DIR" \
        --num-workers 8 "${LIMIT_ARGS[@]}" "$@" 2>&1 | tee "$LOGDIR/$name.log"; then
    echo "!!! ROW $name FAILED (see $LOGDIR/$name.log)"
    if [[ "$KEEP_GOING" == "0" ]]; then
      exit 1
    fi
  fi
}

# Row 1-2: paper protocol reproduction (no balance flags, as in the paper)
run_row paper_resnet50_binary  --task binary     --backbone resnet50 --batch-size 32 --accum-steps 2 --val-metrics-each-epoch
run_row paper_resnet50_attr    --task attribution --backbone resnet50 --batch-size 32 --accum-steps 2 --attr-class-weights --val-metrics-each-epoch

# Row 3: strong multi-task baseline (ConvNeXt-Tiny)
run_row mt_convnext_tiny       --task multitask  --backbone convnext_tiny --batch-size 32 --accum-steps 2 --val-metrics-each-epoch

# Row 4: frequency-aware dual-branch (the paper's headline component)
run_row mt_freq                --task multitask  --backbone convnext_tiny --freq-branch dct --freq-k 32 --batch-size 32 --accum-steps 2 --val-metrics-each-epoch

# Row 5: training-schedule ablation (EMA + label smoothing)
run_row mt_freq_ema_ls         --task multitask  --backbone convnext_tiny --freq-branch dct --freq-k 32 \
                                --ema-decay 0.999 --label-smoothing 0.05 --batch-size 32 --accum-steps 2 --val-metrics-each-epoch

# Row 6: inference-time ablation (TTA + calibrated threshold)
run_row mt_freq_ema_ls_tta     --task multitask  --backbone convnext_tiny --freq-branch dct --freq-k 32 \
                                --ema-decay 0.999 --label-smoothing 0.05 --tta flip_scale --calibrate macro_f1 \
                                --batch-size 32 --accum-steps 2 --val-metrics-each-epoch

# Row 7: foundation-model branch (ViT-L/14 CLIP)
run_row mt_clip                --task multitask  --clip-init clip_vitl14 --freq-branch dct --freq-k 32 \
                                --ema-decay 0.999 --label-smoothing 0.05 --batch-size 16 --accum-steps 4 --val-metrics-each-epoch

echo
echo "=== ablation grid complete (or stopped); aggregate with:"
echo "    $PY scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md"
