"""Distributed helpers: no-ops outside torchrun, real behavior under a gloo group."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mscocoai import distributed


def test_not_distributed_by_default():
    assert not distributed.is_distributed()
    assert distributed.get_rank() == 0
    assert distributed.get_world_size() == 1
    assert distributed.is_main_process()


def test_init_distributed_is_noop_without_env():
    assert distributed.init_distributed() == (0, 1, 0)


def test_all_gather_np_single_process():
    out = distributed.all_gather_np(np.array([1.0, 2.0]))
    assert list(out) == [1.0, 2.0]


def test_broadcast_helpers_single_process():
    assert distributed.broadcast_bool(True) is True
    assert distributed.broadcast_float64(3.25) == pytest.approx(3.25)


def test_build_sampler_plain_dataset_without_distributed(tiny_dataset):
    from mscocoai.data import build_dataset
    ds = build_dataset("train", data_dir=str(tiny_dataset), limit=None, norm="imagenet", img_size=64)
    assert distributed.build_sampler(ds, shuffle=True, seed=1) is None


@pytest.fixture()
def gloo_group():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29599",
                            rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_all_gather_under_gloo(gloo_group):
    out = distributed.all_gather_np(np.array([1.0, 2.0]))
    assert list(out) == [1.0, 2.0]
    assert distributed.is_distributed()
    assert distributed.get_world_size() == 1


def test_all_gather_object_under_gloo(gloo_group):
    out = distributed.all_gather_object({"scores": [1, 2, 3]})
    assert out == [{"scores": [1, 2, 3]}]
