"""Average probabilities of several checkpoints and write a full report.

  python -m mscocoai.ensemble --task binary --checkpoints a.pt b.pt --split test --report-dir reports/ens
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import torch

from mscocoai.constants import LABEL_B_NAMES
from mscocoai.data import build_dataset, build_loader
from mscocoai.metrics import attribution_report, balanced_binary_report, binary_report, per_generator_detection
from mscocoai.models import checkpoint_resize_mode, load_model_for_eval
from mscocoai.tta import average_binary, average_multiclass, tta_views
from mscocoai import report as rep


@torch.no_grad()
def _probs(model, loader, device, task, tta: str = "none", img_size: int = 224):
    model.eval()
    p_det, p_att, ya, yb = [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        views = tta_views(imgs, tta, img_size) if tta != "none" else [imgs]
        det_views, att_views = [], []
        for view in views:
            out = model(view)
            if task == "binary":
                det_views.append(torch.sigmoid(out).squeeze(-1).float())
            elif task == "attribution":
                att_views.append(torch.softmax(out.float(), dim=1))
            else:  # multitask -> contributes to both heads
                z_d, z_a = out
                det_views.append(torch.sigmoid(z_d).float())
                att_views.append(torch.softmax(z_a.float(), dim=1))
        if det_views:
            p_det.append(average_binary(det_views).cpu().numpy())
        if att_views:
            p_att.append(average_multiclass(att_views).cpu().numpy())
        ya.append(batch["label_a"].numpy())
        yb.append(batch["label_b"].numpy())
    ya_ = np.concatenate(ya); yb_ = np.concatenate(yb)
    det = np.concatenate(p_det) if p_det else None
    att = np.concatenate(p_att) if p_att else None
    return det, att, ya_, yb_


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probability-space ensemble of MS-COCOAI checkpoints")
    p.add_argument("--task", required=True, choices=["binary", "attribution", "multitask"])
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", default=None, help="comma-separated weights, e.g. 2,1,1 (default: uniform)")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--norm", default="imagenet", choices=["imagenet", "clip"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--degradation", default="none", choices=["none", "jpeg75", "blur"])
    p.add_argument("--resize-mode", default="auto", choices=["auto", "squash", "canonical"],
                   help="auto: inherit from the checkpoints (all must agree)")
    p.add_argument("--tta", default="none", choices=["none", "flip", "flip_scale"])
    p.add_argument("--holdout-generator", default=None,
                       choices=["sd21", "sdxl", "sd3", "dalle3", "mj6"])
    p.add_argument("--layout", default="auto", choices=["auto", "folder", "parquet"])
    p.add_argument("--report-dir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(args.checkpoints)
    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")], dtype=np.float64)
        if len(w) != n:
            raise SystemExit("--weights length must equal number of checkpoints")
    else:
        w = np.ones(n, dtype=np.float64)
    w = w / w.sum()

    modes = {checkpoint_resize_mode(p) for p in args.checkpoints}
    if len(modes) > 1:
        raise SystemExit(f"checkpoints disagree on resize_mode ({sorted(modes)}); "
                         "ensemble them under one protocol or pass --resize-mode explicitly")
    resize_mode = args.resize_mode
    if resize_mode == "auto":
        resize_mode = modes.pop() if modes else "squash"

    ds = build_dataset(args.split, data_dir=args.data_dir, limit=args.limit,
                       norm=args.norm, img_size=args.img_size, degradation=args.degradation,
                       layout=args.layout, holdout_generator=args.holdout_generator,
                       resize_mode=resize_mode)
    loader = build_loader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, seed=0)

    det_sum = None
    att_sum = None
    det_w = 0.0
    att_w = 0.0
    ya = yb = None
    for i, ckpt_path in enumerate(args.checkpoints):
        model, ckpt = load_model_for_eval(ckpt_path, device=device)
        task = ckpt.get("task") or "binary"
        if args.task == "binary" and task not in ("binary", "multitask"):
            raise SystemExit(f"{ckpt_path} is a {task} checkpoint; expected binary or multitask")
        if args.task == "attribution" and task not in ("attribution", "multitask"):
            raise SystemExit(f"{ckpt_path} is a {task} checkpoint; expected attribution or multitask")
        det, att, ya, yb = _probs(model, loader, device, task, tta=args.tta, img_size=args.img_size)
        if det is not None:
            det_sum = (det * w[i]) if det_sum is None else det_sum + det * w[i]
            det_w += w[i]
        if att is not None:
            att_sum = (att * w[i]) if att_sum is None else att_sum + att * w[i]
            att_w += w[i]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if det_sum is not None and det_w > 0:
        det_sum = det_sum / det_w
    if att_sum is not None and att_w > 0:
        att_sum = att_sum / att_w

    out: dict = {"task": args.task, "checkpoints": list(args.checkpoints),
                 "weights": w.tolist(), "split": args.split,
                 "degradation": args.degradation, "tta": args.tta,
                 "holdout_generator": args.holdout_generator}
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("binary", "multitask"):
        if det_sum is None:
            raise SystemExit("none of the checkpoints produced a detection head")
        out["binary"] = binary_report(ya, det_sum)
        out["binary_balanced"] = balanced_binary_report(ya, det_sum, seed=0)
        out["per_generator_detection"] = per_generator_detection(ya, det_sum, yb)
        rep.save_per_generator_csv(out["per_generator_detection"], report_dir / "per_generator.csv")
    if args.task in ("attribution", "multitask"):
        if att_sum is None:
            raise SystemExit("none of the checkpoints produced an attribution head")
        pred = att_sum.argmax(axis=1)
        out["attribution"] = attribution_report(yb, pred)
        rep.save_confusion_png(out["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix.png")
        rep.save_confusion_png(out["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix_norm.png", normalize=True)

    rep.save_json(out, report_dir / "report.json")
    print(json.dumps(out, indent=2, default=str))
    print(f"[report] ensemble artifacts -> {report_dir}")


if __name__ == "__main__":
    main()
