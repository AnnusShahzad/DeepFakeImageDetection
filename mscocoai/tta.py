"""Deterministic test-time augmentation views and probability averaging."""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def _scale_crop(x: torch.Tensor, scale: float, img_size: int) -> torch.Tensor:
    size = int(round(img_size / scale))
    if size == img_size:
        return x
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    start = (size - img_size) // 2
    return x[..., start:start + img_size, start:start + img_size]


def tta_views(x: torch.Tensor, mode: str, img_size: int = 224) -> List[torch.Tensor]:
    mode = (mode or "none").lower()
    if mode == "none":
        return [x]
    if mode == "flip":
        return [x, torch.flip(x, dims=[-1])]
    if mode == "flip_scale":
        scaled = _scale_crop(x, 0.875, img_size)
        return [x, torch.flip(x, dims=[-1]), scaled, torch.flip(scaled, dims=[-1])]
    raise ValueError(f"unknown --tta mode: {mode}")


def average_binary(probs: List[torch.Tensor]) -> torch.Tensor:
    if len(probs) == 1:
        return probs[0]
    return torch.stack([p.float() for p in probs], dim=0).mean(dim=0)


def average_multiclass(probs: List[torch.Tensor]) -> torch.Tensor:
    if len(probs) == 1:
        return probs[0]
    return torch.stack([p.float() for p in probs], dim=0).mean(dim=0)
