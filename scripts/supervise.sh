#!/usr/bin/env bash
# Restart the whole MS-COCOAI pipeline until it is finished — survives crashes,
# SSH disconnects and power loss (combined with a user @reboot cron entry).
#
#   nohup setsid bash scripts/supervise.sh > outputs/supervisor.log 2>&1 &
#
# Why this is safe:
#   * every training row uses --resume auto, so an interrupted row continues
#     from its last checkpoint instead of starting over;
#   * a row that already wrote results.json is skipped, so completed work is
#     never overwritten (FORCE=1 overrides the skip);
#   * report steps rewrite only their own report directory;
#   * a single flock prevents two supervisors from running at once.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data_mirror}"
export MSCOCOAI_PYTHON="${MSCOCOAI_PYTHON:-python3}"
export FORCE="${FORCE:-0}"

SENTINEL="$REPO_ROOT/outputs/.objectives_complete"
LOCK="$REPO_ROOT/outputs/.supervisor.lock"
mkdir -p "$REPO_ROOT/outputs"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[supervisor] another supervisor holds the lock — exiting"
  exit 0
fi

echo "[supervisor] start $(date -Is)  data=$DATA_DIR"

while [[ ! -f "$SENTINEL" ]]; do
  # Wait until no training is running (an externally started grid may own the GPU).
  while pgrep -f "[m]scocoai.train" >/dev/null 2>&1; do
    echo "[supervisor] waiting for an external training run to finish ($(date -Is))"
    sleep 30
  done

  echo "[supervisor] === ablation grid pass $(date -Is) ==="
  bash scripts/run_ablation.sh --keep-going
  grid_rc=$?

  echo "[supervisor] === objective rows pass $(date -Is) ==="
  bash scripts/run_all_objectives.sh
  obj_rc=$?

  if [[ -f "$SENTINEL" ]]; then
    echo "[supervisor] === pipeline complete ($(date -Is)) ==="
    break
  fi

  echo "[supervisor] pass finished (grid rc=$grid_rc objectives rc=$obj_rc); retry in 60s"
  sleep 60
done

echo "[supervisor] exit $(date -Is)"
