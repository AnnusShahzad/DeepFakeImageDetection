"""CLI: train one MS-COCOAI model.

Examples:
  python -m mscocoai.train --task binary --backbone resnet50 --out-dir outputs/resnet50_binary
  python -m mscocoai.train --task multitask --backbone efficientnet_b0 --out-dir outputs/mt_effb0 \
      --freq-branch dct --ema-decay 0.999 --label-smoothing 0.05
  python -m mscocoai.train --task multitask --clip-init clip_vitl14 --out-dir outputs/mt_clip
  torchrun --nproc_per_node=2 -m mscocoai.train --task binary --backbone resnet50 --out-dir outputs/ddp
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from mscocoai.constants import SEED
from mscocoai.engine import fit


@dataclass
class TrainConfig:
    task: str
    backbone: str
    data_dir: str | None
    limit: int | None
    img_size: int
    norm: str
    batch_size: int
    eval_batch_size: int
    accum_steps: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    early_patience: int
    num_workers: int | str
    seed: int
    out_dir: str
    resume: str                  # auto | none | <path> | weights-only
    balance_loss: bool
    balance_sampler: bool
    attr_class_weights: bool
    label_smoothing: float
    ema_decay: float             # 0 = disabled
    save_steps: int              # 0 = only per-epoch checkpointing
    lambda_d: float
    lambda_a: float
    val_metrics_each_epoch: bool
    drive_backend: str           # none | copy | apps_script
    drive_dir: str | None
    upload_every: str            # epoch | never
    apps_script_url: str | None
    apps_script_token: str | None
    # ---- appended (defaults keep existing constructions valid) ----
    device: str = "auto"         # auto | cpu | cuda | cuda:N | tpu
    data_layout: str = "auto"   # auto | folder | parquet
    max_hours: float = 0.0      # 0 = no wall-clock budget
    min_free_gb: float = 5.0
    upload_metrics: bool = True
    freq_branch: str = "none"   # none | dct | highpass | both
    freq_k: int = 32
    clip_init: str = "none"     # none | clip_vitb16 | clip_vitl14
    clip_unfreeze: str = "last2"  # frozen | last2 | full
    tta: str = "none"           # none | flip | flip_scale
    calibrate: str = "off"      # off | macro_f1 | accuracy
    holdout_generator: str | None = None  # sd21 | sdxl | sd3 | dalle3 | mj6
    channels_last: bool = False
    resume_epoch_only: bool = False

    def as_dict(self):
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-COCOAI trainer")
    p.add_argument("--task", required=True, choices=["binary", "attribution", "multitask"])
    p.add_argument("--backbone", default="resnet50",
                   help="timm model name, e.g. resnet50, efficientnet_b0, efficientnet_b4, convnextv2_base")
    p.add_argument("--data-dir", default=None,
                   help="dataset root: a folder mirror (<root>/{train,validation,test}/<label>/...) "
                        "or an HF parquet dump (<root>/data/<split>-*.parquet); default: download from HF")
    p.add_argument("--data-layout", default="auto", choices=["auto", "folder", "parquet"],
                   help="force the data layout (default: auto-detect)")
    p.add_argument("--holdout-generator", default=None,
                   choices=["sd21", "sdxl", "sd3", "dalle3", "mj6"],
                   help="LOGO protocol: exclude one generator from training and validation, "
                        "evaluate on it (unseen-generator setting)")
    p.add_argument("--limit", type=int, default=None, help="limit samples per split (debug/smoke tests)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--norm", default="imagenet", choices=["imagenet", "clip"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=2, help="gradient accumulation (effective batch = batch*accum)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--early-patience", type=int, default=5)
    p.add_argument("--num-workers", type=str, default="auto",
                   help="dataloader workers; 'auto' chooses a safe value from CPU count")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:N | tpu (default: auto)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", default="auto", help="auto | none | <path-to-checkpoint> | weights-only")
    p.add_argument("--resume-epoch-only", action="store_true",
                   help="on resume, restart the interrupted epoch from scratch instead of skipping batches")
    p.add_argument("--balance-loss", action="store_true", help="pos_weight for binary BCE (1:5 fake:real skew)")
    p.add_argument("--balance-sampler", action="store_true", help="WeightedRandomSampler for attribution/multi-task")
    p.add_argument("--attr-class-weights", action="store_true", help="class-weighted CE for Label_B (paper default for attribution)")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0, help="e.g. 0.999; 0 disables EMA")
    p.add_argument("--save-steps", type=int, default=0, help="extra checkpoint every N optimizer steps (0 = only per epoch)")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="wall-clock budget; stop cleanly (checkpoint + upload + summary) before the limit. "
                        "0 disables it; use --max-hours 11 on Kaggle")
    p.add_argument("--min-free-gb", type=float, default=5.0, help="refuse to start below this free-disk threshold")
    p.add_argument("--freq-branch", default="none", choices=["none", "dct", "highpass", "both"],
                   help="frequency-aware dual-branch component")
    p.add_argument("--freq-k", type=int, default=32, help="size of the retained low-frequency DCT coefficients")
    p.add_argument("--clip-init", default="none", choices=["none", "clip_vitb16", "clip_vitl14"],
                   help="build the model on a CLIP image encoder (ViT-B/16 or ViT-L/14), fine-tuned")
    p.add_argument("--clip-unfreeze", default="last2", choices=["frozen", "last2", "full"],
                   help="which CLIP parameters to train (frozen = head-only; last2 = last 2 blocks + LN + proj)")
    p.add_argument("--tta", default="none", choices=["none", "flip", "flip_scale"],
                   help="test-time augmentation (used by evaluation/TTA, stored in the checkpoint config)")
    p.add_argument("--calibrate", default="off", choices=["off", "macro_f1", "accuracy"],
                   help="fit the binary decision threshold on the validation split (never test)")
    p.add_argument("--channels-last", action="store_true", help="use NHWC memory format on CUDA (faster with tensor cores)")
    # Google Drive sync — no admin needed (see README / guides)
    p.add_argument("--drive-backend", default="none", choices=["none", "copy", "apps_script"])
    p.add_argument("--drive-dir", default=os.environ.get("MSCOCOAI_DRIVE_DIR"),
                   help="destination folder for --drive-backend copy (e.g. a Google Drive for Desktop synced folder)")
    p.add_argument("--upload-every", default="epoch", choices=["epoch", "never"])
    p.add_argument("--upload-metrics", action=argparse.BooleanOptionalAction, default=True,
                   help="also upload history.jsonl / metrics.csv / results.json (default: on)")
    p.add_argument("--apps-script-url", default=os.environ.get("MSCOCOAI_APPS_SCRIPT_URL"),
                   help="web app URL from apps_script/uploader.gs (or env MSCOCOAI_APPS_SCRIPT_URL)")
    p.add_argument("--apps-script-token", default=os.environ.get("MSCOCOAI_APPS_SCRIPT_TOKEN"),
                   help="shared secret from apps_script/uploader.gs (or env MSCOCOAI_APPS_SCRIPT_TOKEN)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.json").open("w") as fh:
        json.dump(cfg.as_dict(), fh, indent=2, default=str)
    print(json.dumps(cfg.as_dict(), indent=2, default=str))
    result = fit(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
