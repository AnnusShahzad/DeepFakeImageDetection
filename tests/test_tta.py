"""TTA: view counts, shape preservation, determinism, probability averaging."""
from __future__ import annotations

import pytest
import torch

from mscocoai import tta


def test_tta_none_returns_identity():
    x = torch.randn(2, 3, 64, 64)
    assert tta.tta_views(x, "none") == [x]


def test_tta_flip_returns_two_views_and_flip_is_reversed():
    x = torch.randn(2, 3, 64, 64)
    views = tta.tta_views(x, "flip")
    assert len(views) == 2
    assert torch.equal(views[0], x)
    assert torch.equal(views[1], torch.flip(x, dims=[-1]))


def test_tta_flip_scale_keeps_shape_and_is_deterministic():
    x = torch.randn(1, 3, 224, 224)
    a = tta.tta_views(x, "flip_scale", img_size=224)
    b = tta.tta_views(x, "flip_scale", img_size=224)
    assert len(a) == 4
    assert all(v.shape == x.shape for v in a)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_average_multiclass_sums_to_one():
    probs = [torch.softmax(torch.randn(4, 6), dim=1),
             torch.softmax(torch.randn(4, 6), dim=1)]
    out = tta.average_multiclass(probs)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)


def test_unknown_mode_raises():
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError):
        tta.tta_views(x, "wat")
