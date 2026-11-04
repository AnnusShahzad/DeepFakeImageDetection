"""Frequency-domain branch: DCT / high-pass cues for synthetic-image detection.

The branch de-normalises its input back to 0..1 pixels (normalisation removes the
per-channel DC term, which would make the DCT meaningless), then computes either
- ``dct``      low-frequency DCT-II coefficients of the grayscale image, and/or
- ``highpass`` the image minus a Gaussian-blurred copy (exposes upsampling residuals).

Both are computed identically at train and eval time, so there is no train/serve skew.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mscocoai.constants import IMAGENET_MEAN, IMAGENET_STD

DCT_CACHE: dict[int, torch.Tensor] = {}


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II basis (n, n), float64 for numerical accuracy."""
    if n in DCT_CACHE:
        return DCT_CACHE[n]
    u = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    x = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    basis = torch.sqrt(torch.tensor(2.0 / n, dtype=torch.float64)) * torch.cos(math.pi * u * (2 * x + 1) / (2 * n))
    basis[0] *= 1.0 / math.sqrt(2.0)
    DCT_CACHE[n] = basis
    return basis


def dct2d(x: torch.Tensor, k: int) -> torch.Tensor:
    """Low-frequency 2D DCT of grayscale (B, 1, H, W) → (B, 1, k, k)."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError("dct2d expects a (B, 1, H, W) grayscale tensor")
    h, w = x.shape[-2:]
    k = int(min(max(1, k), h, w))
    mh = dct_matrix(h).to(device=x.device, dtype=x.dtype)
    mw = dct_matrix(w).to(device=x.device, dtype=x.dtype)
    coeffs = mh @ x @ mw.T
    return coeffs[..., :k, :k]


def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Undo per-channel normalisation (x*std + mean). mean/std may be sequences or tensors."""
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).flatten()
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).flatten()
    return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0, radius: int = 2, eps: float = 1e-8) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding (per channel)."""
    kernel = gaussian_kernel(2 * radius + 1, sigma).to(device=x.device, dtype=x.dtype)
    kernel_x = kernel.view(1, 1, 1, -1).expand(x.size(1), 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1).expand(x.size(1), 1, -1, 1)
    pad = radius
    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel_x, groups=x.size(1))
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    x = F.conv2d(x, kernel_y, groups=x.size(1))
    return x


def highpass(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """x minus a Gaussian-blurred copy — exposes upsampling residuals / fingerprints."""
    return x - gaussian_blur(x, sigma=sigma, radius=2)


class FrequencyBranch(nn.Module):
    """Projects frequency-domain cues to a compact feature vector."""

    def __init__(self, mode: str = "dct", k: int = 32, out_dim: int = 256,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        mode = (mode or "none").lower()
        if mode == "dct":
            self.views = ["dct"]
        elif mode == "highpass":
            self.views = ["highpass"]
        elif mode == "both":
            self.views = ["dct", "highpass"]
        else:
            raise ValueError(f"unknown frequency mode: {mode}")
        self.mode = mode
        self.k = int(k)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        def tower() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, out_dim),
            )

        if "dct" in self.views:
            self.dct_tower = tower()
        if "highpass" in self.views:
            self.high_tower = tower()
        self.proj = nn.Linear(len(self.views) * out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = denormalize(x, self.mean, self.std).clamp(0.0, 1.0)
        gray = raw.mean(dim=1, keepdim=True)
        feats: list[torch.Tensor] = []
        if "dct" in self.views:
            d = dct2d(gray, self.k)
            d = torch.log1p(d.abs())
            # per-sample normalisation of the coefficient map for scale invariance
            norm = d.flatten(1).max(dim=1).values.clamp_min(1e-6).reshape(-1, 1, 1, 1)
            feats.append(self.dct_tower(d / norm))
        if "highpass" in self.views:
            h = highpass(gray, sigma=1.0)
            norm = h.flatten(1).abs().max(dim=1).values.clamp_min(1e-6).reshape(-1, 1, 1, 1)
            feats.append(self.high_tower(h / norm))
        return self.proj(torch.cat(feats, dim=1))
