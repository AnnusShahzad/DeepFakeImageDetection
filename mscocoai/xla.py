"""TPU device support via torch_xla.

NOTE: this environment had no TPU available, so this path is **untested on real
TPU hardware**. It is written to torch_xla's documented API and kept behind
``--device tpu``. If it is wrong, the failure will be loud, never silent.
"""
from __future__ import annotations

import contextlib
from typing import Any


def xla_available() -> bool:
    try:
        import torch_xla  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False


def xla_device() -> Any:
    import torch_xla
    return torch_xla.device()


def xla_seed(seed: int) -> None:
    if not xla_available():
        return
    try:
        from torch_xla.utils.utils import xla_seed as _xla_seed
        _xla_seed(seed)
    except Exception:  # noqa: BLE001 — seeding must never break setup
        try:
            torch_xla = __import__("torch_xla")
            torch_xla.manual_seed(seed)
        except Exception:
            pass


def xla_mark_step() -> None:
    if not xla_available():
        return
    import torch_xla.core.xla_model as xm
    xm.mark_step()


def xla_amp_context():
    """bf16 autocast on XLA when available, else a no-op context."""
    try:
        import torch
        return torch.autocast("xla", dtype=torch.bfloat16)
    except Exception:  # noqa: BLE001 — never crash outside a TPU
        return contextlib.nullcontext()
