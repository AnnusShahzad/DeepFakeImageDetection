import os

import pytest
import torch

from mscocoai import models
from mscocoai.models import (
    CheckpointCorrupt, EMA, MultiTaskEffB0, atomic_torch_save, build_backbone,
    build_model, export_fp16, find_latest_valid_checkpoint, load_checkpoint,
    save_checkpoint,
)


def test_cnn_forward_shape():
    model = build_backbone("resnet18", num_classes=1, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1)


def test_multitask_shapes():
    model = MultiTaskEffB0("efficientnet_b0", pretrained=False).eval()
    with torch.no_grad():
        zd, za = model(torch.randn(2, 3, 64, 64))
    assert zd.shape == (2,) and za.shape == (2, 6)


def test_checkpoint_roundtrip(tmp_path):
    model = build_backbone("resnet18", num_classes=1, pretrained=False)
    path = tmp_path / "last.pt"
    save_checkpoint({"model": model.state_dict(), "config": {"task": "binary"}, "task": "binary",
                     "backbone": "resnet18", "epoch": 3, "best_val": 0.5}, path)
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 3 and ckpt["task"] == "binary"
    for k, v in model.state_dict().items():
        assert torch.equal(ckpt["model"][k], v)


def test_corrupt_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointCorrupt):
        load_checkpoint(path)


def test_find_latest_valid_falls_back_to_bak(tmp_path):
    model = build_backbone("resnet18", num_classes=1, pretrained=False)
    good = tmp_path / "last.pt.bak"
    save_checkpoint({"model": model.state_dict(), "config": {}, "task": "binary",
                     "backbone": "resnet18"}, good)
    (tmp_path / "last.pt").write_bytes(b"garbage")
    (tmp_path / "LATEST").write_text("last.pt")
    found = find_latest_valid_checkpoint(tmp_path)
    assert found == good


def test_fp16_export(tmp_path):
    model = build_backbone("resnet18", num_classes=1, pretrained=False)
    src = tmp_path / "full.pt"
    dst = tmp_path / "half.pt"
    save_checkpoint({"model": model.state_dict(), "config": {}, "task": "binary",
                     "backbone": "resnet18", "epoch": 5, "val_metrics": {"accuracy": 0.9}}, src)
    export_fp16(src, dst)
    ckpt = load_checkpoint(dst)
    float_vals = [v for v in ckpt["model"].values() if v.is_floating_point()]
    assert float_vals and all(v.dtype == torch.float16 for v in float_vals)
    assert ckpt["epoch"] == 5 and ckpt["config"] == {}


def test_ema_update_moves_toward_params():
    model = torch.nn.Linear(4, 1)
    ema = EMA(model, decay=0.5)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for k, v in ema.shadow.items():
        assert torch.any(v != before[k])


def test_model_spec_roundtrip_rebuilds_same_architecture():
    spec = models.model_spec("multitask", "resnet18", freq_branch="dct", freq_k=16)
    m1 = models.build_model_from_spec(spec)
    m2 = models.build_model_from_spec(spec)
    assert set(m1.state_dict()) == set(m2.state_dict())


def test_load_model_for_eval_uses_model_spec(tmp_path):
    model = models.build_model("multitask", "resnet18", pretrained=False,
                               freq_branch="dct", freq_k=16)
    spec = models.model_spec("multitask", "resnet18", freq_branch="dct", freq_k=16)
    path = tmp_path / "ck.pt"
    models.save_checkpoint({"model": model.state_dict(), "config": {}, "task": "multitask",
                            "backbone": "resnet18", "model_spec": spec}, path)
    loaded, ckpt = models.load_model_for_eval(path)
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_load_model_for_eval_falls_back_to_legacy_checkpoint(tmp_path):
    model = models.build_backbone("resnet18", num_classes=1, pretrained=False)
    path = tmp_path / "legacy.pt"
    models.save_checkpoint({"model": model.state_dict(), "config": {}, "task": "binary",
                            "backbone": "resnet18"}, path)
    loaded, ckpt = models.load_model_for_eval(path)
    assert loaded(torch.randn(2, 3, 64, 64)).shape == (2, 1)


def test_clip_task_model_builds_with_monkeypatched_encoder(monkeypatch):
    import mscocoai.models as m

    class _FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = type("V", (), {"output_dim": 32})()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def encode_image(self, x):
            return torch.randn(x.shape[0], 32)

    monkeypatch.setattr(m, "_create_open_clip", lambda name, pretrained: _FakeClip())
    model = m.build_model("multitask", "resnet18", pretrained=False,
                          clip_init="clip_vitb16", clip_unfreeze="frozen")
    zd, za = model(torch.randn(2, 3, 64, 64))
    assert zd.shape == (2,) and za.shape == (2, 6)


def test_clip_unfreeze_freezes_encoder_by_default(monkeypatch):
    import mscocoai.models as m

    class _FakeClip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = type("V", (), {"output_dim": 16})()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def encode_image(self, x):
            return torch.randn(x.shape[0], 16)

    monkeypatch.setattr(m, "_create_open_clip", lambda name, pretrained: _FakeClip())
    model = m.ClipTaskModel("binary", "ViT-B-16", "openai", unfreeze="frozen")
    assert all(not p.requires_grad for p in model.encoder.parameters())
