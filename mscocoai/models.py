"""Model builders, crash-safe checkpoint I/O, self-describing checkpoints, EMA."""
from __future__ import annotations

import re

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm

from mscocoai.constants import IMAGENET_MEAN, IMAGENET_STD, NUM_ATTR_CLASSES
from mscocoai.frequency import FrequencyBranch


class CheckpointCorrupt(RuntimeError):
    pass


# ------------------------------------------------------------------ backbones
def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Any timm model name; num_classes=1 gives a single logit (binary head)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


class MultiTaskNet(nn.Module):
    """Shared backbone + binary (1-logit) head + 6-class attribution head.

    Supports an optional frequency-domain branch (DCT / high-pass / both),
    concatenated with the backbone features before the two heads.
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 hidden_dropout: float = 0.2, freq_branch: str = "none",
                 freq_k: int = 32, freq_dim: int = 256,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.features = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.features.num_features
        self.freq = None
        self.freq_dim = 0
        self.freq_branch = (freq_branch or "none").lower()
        if self.freq_branch != "none":
            self.freq = FrequencyBranch(self.freq_branch, k=freq_k, out_dim=freq_dim,
                                        mean=mean, std=std)
            self.freq_dim = freq_dim
        in_dim = feat_dim + self.freq_dim
        self.dropout = nn.Dropout(hidden_dropout)
        self.det_head = nn.Linear(in_dim, 1)
        self.attr_head = nn.Linear(in_dim, NUM_ATTR_CLASSES)

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        if self.freq is not None:
            h = torch.cat([h, self.freq(x)], dim=1)
        h = self.dropout(h)
        return self.det_head(h).squeeze(-1), self.attr_head(h)


class MultiTaskEffB0(MultiTaskNet):
    """Legacy name: EfficientNet-B0 shared backbone + two heads (paper O4)."""

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 hidden_dropout: float = 0.2):
        super().__init__(backbone=backbone, pretrained=pretrained,
                         hidden_dropout=hidden_dropout, freq_branch="none")


def _create_open_clip(model_name: str, pretrained) -> nn.Module:
    """Wrapper so tests can monkeypatch the encoder creation."""
    import open_clip
    if pretrained is None:
        # random-init weights with correct shapes (no download)
        return open_clip.create_model(model_name, pretrained=None)
    import open_clip as oc
    m, _, _ = oc.create_model_and_transforms(model_name, pretrained=pretrained)
    return m


_CLIP_PRESETS = {
    "clip_vitb16": ("ViT-B-16", "openai"),
    "clip_vitl14": ("ViT-L-14", "openai"),
}


def _clip_feat_dim(encoder: nn.Module) -> int:
    visual = getattr(encoder, "visual", None)
    for obj in (visual, encoder):
        if obj is None:
            continue
        for attr in ("output_dim",):
            val = getattr(obj, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    # fallback: run a one-sample forward on CPU (one-off, small cost)
    device = next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        out = encoder.encode_image(torch.zeros(1, 3, 224, 224, device=device))
    encoder.train(was_training)
    return int(out.shape[-1])


def _apply_unfreeze(encoder: nn.Module, unfreeze: str) -> None:
    unfreeze = (unfreeze or "last2").lower()
    if unfreeze == "full":
        for p in encoder.parameters():
            p.requires_grad_(True)
        encoder.train()
        return
    if unfreeze == "frozen":
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
        return
    # "last2" — unfreeze the last two transformer blocks + ln_post + proj
    for p in encoder.parameters():
        p.requires_grad_(False)
    block_ids = []
    for name, _ in encoder.named_parameters():
        m = re.search(r"resblocks\.(\d+)", name)
        if m:
            block_ids.append(int(m.group(1)))
    if block_ids:
        keep = {max(block_ids), max(block_ids) - 1}
    else:
        keep = set()
    for name, p in encoder.named_parameters():
        if any(f"resblocks.{i}." in name for i in keep) \
                or "ln_post" in name \
                or "proj" in name \
                or "text_projection" in name:
            p.requires_grad_(True)
    encoder.train()


class ClipTaskModel(nn.Module):
    """CLIP image encoder + detection/attribution head(s).

    task: binary | attribution | multitask
    """

    def __init__(self, task: str, model_name: str, pretrained,
                 unfreeze: str = "last2", hidden_dropout: float = 0.2):
        super().__init__()
        self.task = task
        self.encoder = _create_open_clip(model_name, pretrained)
        _apply_unfreeze(self.encoder, unfreeze)
        feat_dim = _clip_feat_dim(self.encoder)
        self.dropout = nn.Dropout(hidden_dropout)
        if task == "binary":
            self.head_d = nn.Linear(feat_dim, 1)
        elif task == "attribution":
            self.head_a = nn.Linear(feat_dim, NUM_ATTR_CLASSES)
        elif task == "multitask":
            self.head_d = nn.Linear(feat_dim, 1)
            self.head_a = nn.Linear(feat_dim, NUM_ATTR_CLASSES)
        else:
            raise ValueError(f"unknown task: {task}")
        self.clip_unfreeze = unfreeze

    def features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder.encode_image(x)
        f = f.float()
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor):
        f = self.dropout(self.features(x))
        if self.task == "binary":
            return self.head_d(f).squeeze(-1)
        if self.task == "attribution":
            return self.head_a(f)
        return self.head_d(f).squeeze(-1), self.head_a(f)


def build_model(task: str, backbone: str, pretrained: bool = True, *,
                freq_branch: str = "none", freq_k: int = 32,
                clip_init: str = "none", clip_unfreeze: str = "last2",
                img_size: int = 224, mean=IMAGENET_MEAN,
                std=IMAGENET_STD) -> nn.Module:
    if clip_init and clip_init != "none":
        model_name, tag = _CLIP_PRESETS[clip_init]
        return ClipTaskModel(task, model_name, tag if pretrained else None,
                             unfreeze=clip_unfreeze)
    if task == "binary":
        return build_backbone(backbone, num_classes=1, pretrained=pretrained)
    if task == "attribution":
        return build_backbone(backbone, num_classes=NUM_ATTR_CLASSES, pretrained=pretrained)
    if task == "multitask":
        return MultiTaskNet(backbone=backbone, pretrained=pretrained,
                            freq_branch=freq_branch, freq_k=freq_k,
                            mean=mean, std=std)
    raise ValueError(f"unknown task: {task}")


# ------------------------------------------------------- self-describing spec
def model_spec(task: str, backbone: str, *, freq_branch: str = "none", freq_k: int = 32,
               clip_init: str = "none", clip_unfreeze: str = "last2",
               img_size: int = 224, norm: str = "imagenet",
               mean=IMAGENET_MEAN, std=IMAGENET_STD) -> dict[str, Any]:
    return {
        "task": task,
        "backbone": backbone,
        "freq_branch": (freq_branch or "none").lower(),
        "freq_k": int(freq_k),
        "clip_init": (clip_init or "none").lower(),
        "clip_unfreeze": (clip_unfreeze or "last2").lower(),
        "img_size": int(img_size),
        "norm": norm,
        "mean": list(mean),
        "std": list(std),
    }


def build_model_from_spec(spec: dict[str, Any] | None, pretrained: bool = False) -> nn.Module:
    if not spec:
        raise CheckpointCorrupt("checkpoint does not contain a model_spec")
    spec = dict(spec)
    mean = tuple(spec.get("mean") or IMAGENET_MEAN)
    std = tuple(spec.get("std") or IMAGENET_STD)
    return build_model(
        spec.get("task", "binary"),
        spec.get("backbone", "resnet50"),
        pretrained=pretrained,
        freq_branch=spec.get("freq_branch", "none"),
        freq_k=int(spec.get("freq_k", 32)),
        clip_init=spec.get("clip_init", "none"),
        clip_unfreeze=spec.get("clip_unfreeze", "last2"),
        img_size=int(spec.get("img_size", 224)),
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------- checkpoints
def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    buf_path = Path(path)
    with tempfile.NamedTemporaryFile(dir=str(buf_path.parent), prefix=buf_path.name + ".tmp", delete=False) as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
        tmp_name = fh.name
    os.replace(tmp_name, buf_path)


def save_checkpoint(state: dict, path: str | os.PathLike) -> None:
    atomic_torch_save(state, path)


def load_checkpoint(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> dict:
    path = str(path)
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 — convert any load failure into a clear error
        raise CheckpointCorrupt(f"Could not load checkpoint: {path} ({exc})") from exc
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise CheckpointCorrupt(f"Malformed checkpoint (missing 'model' state dict): {path}")
    return ckpt


def find_latest_valid_checkpoint(out_dir: str | os.PathLike) -> Path | None:
    """Return the newest loadable checkpoint in out_dir, or None.

    Prefers files listed in LATEST (written alongside every save); falls back
    to scanning for last.pt / last.pt.bak / any *.pt ordered by mtime.
    """
    out_dir = Path(out_dir)
    candidates: list[Path] = []
    marker = out_dir / "LATEST"
    if marker.is_file():
        try:
            name = marker.read_text().strip()
            if name:
                candidates.append(out_dir / name)
        except Exception:  # noqa: BLE001
            pass
    candidates += [out_dir / "last.pt", out_dir / "last.pt.bak", out_dir / "best.pt"]
    candidates += sorted(out_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    for cand in candidates:
        if cand in seen or not cand.is_file():
            continue
        seen.add(cand)
        try:
            load_checkpoint(cand, map_location="cpu")
            return cand
        except CheckpointCorrupt:
            continue
    return None


def rotate_backups(out_dir: str | os.PathLike, target_name: str = "last.pt") -> None:
    """Keep last.pt.bak as one previous good copy before overwriting last.pt."""
    out_dir = Path(out_dir)
    src = out_dir / target_name
    if src.is_file():
        try:
            os.replace(src, out_dir / (target_name + ".bak"))
        except OSError:
            pass


def export_fp16(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    ckpt = load_checkpoint(src, map_location="cpu")
    model_sd = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in ckpt["model"].items()}
    slim = {"model": model_sd,
            "config": ckpt.get("config", {}),
            "epoch": ckpt.get("epoch", -1),
            "step_in_epoch": ckpt.get("step_in_epoch"),
            "task": ckpt.get("task"),
            "backbone": ckpt.get("backbone"),
            "model_spec": ckpt.get("model_spec"),
            "threshold": ckpt.get("threshold"),
            "val_metrics": ckpt.get("val_metrics", {})}
    atomic_torch_save(slim, dst)


class EMA:
    """Exponential moving average of model parameters (decay=0.999 typical)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def store(self, model: nn.Module) -> dict:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k in msd:
            msd[k].copy_(self.shadow[k].to(msd[k].dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict) -> None:
        msd = model.state_dict()
        for k in msd:
            msd[k].copy_(backup[k])


def load_model_for_eval(checkpoint: str | os.PathLike, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = load_checkpoint(checkpoint, map_location=device)
    spec = ckpt.get("model_spec")
    if spec:
        try:
            model = build_model_from_spec(spec, pretrained=False)
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()
            return model, ckpt
        except CheckpointCorrupt:
            pass
    # legacy fallback: rebuild from task/backbone
    cfg = ckpt.get("config") or {}
    task = ckpt.get("task") or cfg.get("task") or "binary"
    backbone = ckpt.get("backbone") or cfg.get("backbone") or "resnet50"
    model = build_model(task, backbone, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
