#!/usr/bin/env bash
# Complete OBJECTIVE.md (O1-O5): the missing paper rows + frozen-CLIP probe +
# the *single* test-set read per checkpoint + aggregation into results/.
#
# Queues BEHIND the current ablation grid (never competes for the GPU).
# Safe to re-run: completed training rows are skipped, interrupted rows resume
# from their last checkpoint (--resume auto), report steps re-write their own
# outputs.
#
#   DATA_DIR=~/mscocoai/data_mirror MSCOCOAI_PYTHON=~/venvs/mscocoai/bin/python \
#       nohup bash scripts/run_all_objectives.sh > ~/mscocoai/objectives_full.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${MSCOCOAI_PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data_mirror}"
# Batch 64 x accum 1 keeps the effective batch of 64 used by the grid rows but
# halves the number of forward/backward launches: at batch 32 the A4000 was
# launch-bound (97% "util" at only 117 W of 300 W).
BATCH="${BATCH:-64}"
ACCUM="${ACCUM:-1}"
EVAL_BATCH="${EVAL_BATCH:-128}"
WORKERS="${WORKERS:-12}"

echo "=== waiting for the current ablation grid to exit ==="
while pgrep -f "[r]un_ablation.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== grid done; starting objective completion ==="

run() {
  echo
  echo ">>> $*"
  # A failing step must not kill the queue: the supervisor retries the pass.
  "$@" 2>&1 | tee -a "outputs/objectives_driver.log" || echo "!! step failed: $*"
}

train_row() {
  local name="$1"; shift
  if [[ -f "outputs/$name/results.json" && "${FORCE:-0}" != "1" ]]; then
    echo
    echo ">>> $name already complete — skipping (FORCE=1 to retrain)"
    return 0
  fi
  run "$PY" -m mscocoai.train --out-dir "outputs/$name" --data-dir "$DATA_DIR" \
      --batch-size "$BATCH" --accum-steps "$ACCUM" --eval-batch-size "$EVAL_BATCH" \
      --num-workers "$WORKERS" --resume auto "$@"
}

mkdir -p outputs results

########## Shortcut control: metadata-only baseline (no pixels) ##########
run "$PY" scripts/shortcut_baseline.py --data-dir "$DATA_DIR" --out-dir results/shortcut

########## Shortcut control: canonical-resize protocol (resolution cue removed) ##########
# Same models as the headline rows, trained on the identical two-step resize for
# every image. The gap to the standard rows is the size of the dataset's
# native-resolution shortcut.
train_row canonical_resnet50_binary --task binary --backbone resnet50 \
    --balance-loss --val-metrics-each-epoch --resize-mode canonical
train_row canonical_mt_freq --task multitask --backbone convnext_tiny --freq-branch dct --freq-k 32 \
    --val-metrics-each-epoch --resize-mode canonical

########## O1 / O3 — EfficientNet-B0 (paper protocol, second CNN) ##########
train_row effb0_binary --task binary --backbone efficientnet_b0 \
    --balance-loss --val-metrics-each-epoch
train_row effb0_attr --task attribution --backbone efficientnet_b0 \
    --attr-class-weights --val-metrics-each-epoch

########## O4 — paper-protocol multi-task (EffB0, lambda_d = lambda_a = 1) ##########
train_row mt_effb0 --task multitask --backbone efficientnet_b0 \
    --balance-loss --attr-class-weights --val-metrics-each-epoch

########## O5 — LOGO: detectors that never saw a generator ##########
# Trained with the generator removed from train *and* validation; the test
# report's per_generator.csv then shows how the model handles it unseen.
train_row logo_binary_sd21 --task binary --backbone resnet50 \
    --balance-loss --val-metrics-each-epoch --holdout-generator sd21
train_row logo_binary_sd3 --task binary --backbone resnet50 \
    --balance-loss --val-metrics-each-epoch --holdout-generator sd3

########## O2 — frozen CLIP ViT-B/16 + linear probe (UniFD protocol) ##########
run "$PY" -m mscocoai.clip_probe --mode extract --data-dir "$DATA_DIR" --out-dir outputs/clip_vitb16
run "$PY" -m mscocoai.clip_probe --mode train   --task binary     --out-dir outputs/clip_vitb16 --balance-loss
run "$PY" -m mscocoai.clip_probe --mode train   --task attribution --out-dir outputs/clip_vitb16 --attr-class-weights

########## Final test-split evaluations (test read exactly once per report) ##########
for run_name in paper_resnet50_binary paper_resnet50_attr \
                effb0_binary effb0_attr mt_effb0 \
                mt_convnext_tiny mt_freq mt_freq_ema_ls mt_freq_ema_ls_tta mt_clip \
                canonical_resnet50_binary canonical_mt_freq \
                logo_binary_sd21 logo_binary_sd3; do
  ck="outputs/${run_name}/best.pt"
  [ -f "$ck" ] || { echo "!! missing $ck — skipping"; continue; }
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --num-workers "$WORKERS" --report-dir "reports/${run_name}_test" \
      || echo "!! eval failed for $run_name"
done

########## O2 probe test reports ##########
run "$PY" -m mscocoai.clip_probe --mode eval --task binary      --out-dir outputs/clip_vitb16 --report-dir reports/clip_binary_test
run "$PY" -m mscocoai.clip_probe --mode eval --task attribution --out-dir outputs/clip_vitb16 --report-dir reports/clip_attr_test

########## Robustness probes (JPEG-75 / blur) for the two headline binary models ##########
for run_name in paper_resnet50_binary mt_freq; do
  ck="outputs/${run_name}/best.pt"
  [ -f "$ck" ] || continue
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --degradation jpeg75 --num-workers "$WORKERS" --no-samples \
      --report-dir "reports/${run_name}_jpeg75"
  run "$PY" -m mscocoai.evaluate --checkpoint "$ck" --data-dir "$DATA_DIR" \
      --split test --degradation blur --num-workers "$WORKERS" --no-samples \
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
run "$PY" scripts/collect_results.py outputs/* --csv results/ablation.csv --md results/ablation.md --reports reports

touch outputs/.objectives_complete
echo
echo "=== DONE. See results/ablation.md for the full table and reports/ for per-model artifacts."
