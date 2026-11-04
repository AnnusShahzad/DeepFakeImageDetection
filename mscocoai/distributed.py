"""Distributed training helpers (torchrun / DDP).

The single-process path is a no-op: ``init_distributed`` returns (0, 1, 0)
unless RANK/WORLD_SIZE are set by the launcher.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialise the process group when launched with torchrun.

    Returns (rank, world_size, local_rank). In a single-process run, returns
    (0, 1, 0) and initialises nothing.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = get_local_rank()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size > 1 and not is_distributed():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def build_sampler(dataset, *, shuffle: bool, seed: int = 0, drop_last: bool = False):
    """DistributedSampler when distributed, else None (caller uses regular shuffle)."""
    from torch.utils.data import DistributedSampler
    if is_distributed():
        return DistributedSampler(dataset, num_replicas=get_world_size(),
                                  rank=get_rank(), shuffle=shuffle,
                                  seed=seed, drop_last=drop_last)
    return None


def _collective_tensor(value, dtype=torch.float64) -> torch.Tensor:
    """Tensor on the right device for the active backend (nccl needs CUDA)."""
    t = torch.tensor([value], dtype=dtype)
    if is_distributed() and dist.get_backend() == "nccl" and torch.cuda.is_available():
        t = t.cuda()
    return t


def all_gather_np(arr: np.ndarray) -> np.ndarray:
    """All-gather a 1-D/N-D numpy array from every rank, concatenated on axis 0."""
    arr = np.ascontiguousarray(arr)
    if not is_distributed():
        return arr
    t = torch.from_numpy(arr).cpu()
    if dist.get_backend() == "nccl":
        t = t.cuda()
    gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(gathered, t)
    return np.concatenate([g.cpu().numpy() for g in gathered], axis=0)


def all_reduce_sum(value: float) -> float:
    t = _collective_tensor(float(value), torch.float64)
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_reduce_max_int(value: int) -> int:
    t = _collective_tensor(int(value), torch.int64)
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def all_gather_object(obj):
    """All-gather arbitrary Python objects (uneven sizes OK)."""
    if not is_distributed():
        return [obj]
    out = [None] * get_world_size()
    dist.all_gather_object(out, obj)
    return out


def broadcast_bool(value: bool, src: int = 0) -> bool:
    """Broadcast a boolean decision so all ranks stop/continue together."""
    t = torch.tensor([int(value)], dtype=torch.int64)
    if is_distributed():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            t = t.cuda()
        dist.broadcast(t, src=src)
    return bool(t.item())


def broadcast_float64(value: float, src: int = 0) -> float:
    """Broadcast a float so every rank compares the same bits (no fp drift)."""
    t = torch.tensor([float(value)], dtype=torch.float64)
    if is_distributed():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            t = t.cuda()
        dist.broadcast(t, src=src)
    return float(t.item())
