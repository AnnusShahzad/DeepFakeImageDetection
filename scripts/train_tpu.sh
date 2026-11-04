#!/usr/bin/env bash
# Launch training on a Kaggle TPU runtime (TPU v3-8).
#
# Requirements on the notebook:
#   Runtime -> Change runtime type -> TPU v3-8
#   Internet: ON
#   Then: pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
#
# NOTE: the TPU path is untested on real TPU hardware (none was available during
# development). It is written to torch_xla's documented API; if it is wrong, the
# failure will be loud, not silent.
set -euo pipefail

if ! python3 -c "import torch_xla" 2>/dev/null; then
    echo "torch_xla is not installed."
    echo "On Kaggle TPU: pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html"
    exit 1
fi

python3 -m mscocoai.train --device tpu "$@"
