#!/usr/bin/env bash
# Complete OBJECTIVE.md (O1-O5): the missing paper rows + frozen-CLIP probe +
# the *single* test-set read per checkpoint + aggregation into results/.
#
# Queues BEHIND the current ablation grid (never competes for the GPU).
# Safe to re-run: training uses --resume auto; report steps re-write their outputs.
#
#   DATA_DIR=~/mscocoai/data_mirror MSCOCOAI_PYTHON=~/venvs/mscocoai/bin/python \
#       nohup bash scripts/run_all_objectives.sh > ~/mscocoai/objectives_full.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${MSCOCOAI_PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data_mirror}"

echo "=== waiting for the current ablation grid to exit ==="
while pgrep -f "[r]un_ablation.sh" >/dev/null; do sleep 60; done
echo "=== grid done; starting objective completion ==="

run() {
  echo
  echo ">>> $*"
  "$@" 2>&1 | tee -a "outputs/objectives_driver.log"
}
mkdir -p outputs

########## O1 / O3 — EfficientNet-B0 (paper protocol, second CNN) ##########
run "$PY" -m mscocoai.train --task binary --backbone efficientnet_b0 \
    --data-dir "$DATA_DIR" --out-dir outputs/effb0_binary --resume auto \
    --balance-loss --val-metrics-each-epoch --num-workers 8
run "$PY" -m mscocoai.train --task attribution --backbone efficientnet_b0 \
    --data-dir "$DATA_DIR" --out-dir outputs/effb0_attr --resume auto \
    --attr-class-weights --val-metrics-each-epoch --num-workers 8

########## O4 — paper-protocol multi-task (EffB0, lambda_d = lambda_a = 1) ##########
run "$PY" -m mscocoai.train --task multitask --backbone efficientnet_b0 \
    --data-dir "$DATA_DIR" --out-dir outputs/mt_effb0 --resume auto \
    --balance-loss --attr-class-weights --val-metrics-each-epoch --num-workers 8

########## O2 — frozen CLIP ViT-B/16 + linear probe (UniFD protocol) ##########
run "$PY" -m mscocoai.clip_probe --mode extract --data-dir "$DATA_DIR" --out-dir outputs/clip_vitb16
run "$PY" -m mscocoai.clip_probe --mode train   --task binary     --out-dir outputs/clip_vitb16 --balance-loss
run "$PY" -m mscocoai.clip_probe --mode train   --task attribution --out-dir outputs/clip_vitb16 --attr-class-weights

########## Final test-split evaluations (test read exactly once per report) ##########
EVAL_OUT=outputs
for run_name in paper_resnet50_binary paper_resnet50_attr \
                effb0_binary effb0_attr mt_effb0 \
                mt_convnext_tiny mt_freq mt_freq_ema_ls mt_freq_ema_ls_tta mt_clip; do
  ck="outputs/${run_name}/best.pt"
  [ -f "$ck" ] || { echo "!! missing $ck — skipping"; continue; }
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --num-workers 8 --report-dir "reports/${run_name}_test" \
      || echo "!! eval failed for $run_name"
done

########## O2 probe test reports ##########
run "$PY" -m mscocoai.clip_probe --mode eval --task binary      --out-dir outputs/clip_vitb16 --report-dir reports/clip_binary_test \
    || echo "!! clip binary eval failed"
run "$PY" -m mscocoai.clip_probe --mode eval --task attribution --out-dir outputs/clip_vitb16 --report-dir reports/clip_attr_test \
    || echo "!! clip attribution eval failed"

########## Robustness probes (JPEG-75 / blur) for the two headline binary models ##########
for run_name in paper_resnet50_binary mt_freq; do
  ck="outputs/${run_name}/best.pt"
  [ -f "$ck" ] || continue
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --degradation jpeg75 --num-workers 8 --no-samples \
      --report-dir "reports/${run_name}_jpeg75"
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --degradation blur --num-workers 8 --no-samples \
      --report-dir "reports/${run_name}_blur"
done

########## Optional ensemble comparison (documentation only) ##########
run "$PY" -m mscocoai.ensemble --task binary \
    --checkpoints outputs/paper_resnet50_binary/best.pt outputs/mt_freq/best.pt \
    --data-dir "$DATA_DIR" --split test --report-dir reports/ensemble_binary
run "$PY" -m mscocoai.ensemble --task attribution \
    --checkpoints outputs/paper_resnet50_attr/best.pt outputs/mt_freq/best.pt \
    --data-dir "$DATA_DIR" --split test --report-dir reports/ensemble_attr

########## Aggregate everything into results/ ##########
mkdir -p results
run "$PY" scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md
echo
echo "=== DONE. See results/ablation.md for the full table and reports/ for per-model artifacts."
