#!/usr/bin/env bash
# Launch torchrun for multi-GPU training (Kaggle T4 x2, or your own multi-GPU box).
# Usage: bash scripts/train_ddp.sh [torchrun-args...] -m mscocoai.train --task binary ...
set -euo pipefail

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[train-ddp] torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} $*"
exec torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" -m mscocoai.train "$@"
