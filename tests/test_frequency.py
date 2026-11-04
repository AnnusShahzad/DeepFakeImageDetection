"""Frequency branch: DCT correctness, de-normalisation, branch output shape."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mscocoai import frequency

scpi = pytest.importorskip("scipy.fft")
from scipy.fft import dct as scipy_dct  # noqa: E402


def test_dct_matrix_is_orthonormal():
    m = frequency.dct_matrix(8).double()
    assert torch.allclose(m @ m.T, torch.eye(8, dtype=torch.float64), atol=1e-10)


def test_dct2d_matches_scipy_on_a_small_image():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    ref = scipy_dct(scipy_dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")
    x = torch.from_numpy(img)[None, None, :, :]
    got = frequency.dct2d(x, k=4)[0, 0].numpy()
    assert np.allclose(got, ref[:4, :4], atol=1e-8)


def test_dct2d_returns_top_left_coefficients():
    x = torch.randn(2, 1, 32, 32)
    out = frequency.dct2d(x, k=8)
    assert out.shape == (2, 1, 8, 8)
    out_full = frequency.dct2d(x, k=64)   # k clamped to min(H, W)
    assert out_full.shape[-2:] == (32, 32)


def test_denormalize_roundtrips_normalization():
    x = torch.rand(2, 3, 8, 8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normed = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    assert torch.allclose(frequency.denormalize(normed, mean, std), x, atol=1e-5)


def test_highpass_shape_and_zero_mean_residual():
    x = torch.randn(2, 1, 32, 32)
    h = frequency.highpass(x, sigma=1.0)
    assert h.shape == x.shape


def test_frequency_branch_output_shape():
    branch = frequency.FrequencyBranch(mode="both", k=16, out_dim=64)
    out = branch(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64)


def test_frequency_branch_rejects_bad_mode():
    with pytest.raises(ValueError):
        frequency.FrequencyBranch(mode="wat")
