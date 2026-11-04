"""Evaluate a checkpoint on a split (default: held-out test) and write reports.

The test split is only read here — never during model selection. Typical final run:

  python -m mscocoai.evaluate --checkpoint outputs/resnet50_binary/best.pt --split test --report-dir reports/resnet50_binary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mscocoai.constants import LABEL_B_NAMES
from mscocoai.data import build_dataset, build_loader
from mscocoai.metrics import attribution_report, binary_report, per_generator_detection
from mscocoai.models import load_model_for_eval
from mscocoai import report as rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-COCOAI evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--norm", default="imagenet", choices=["imagenet", "clip"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--degradation", default="none", choices=["none", "jpeg75", "blur"],
                   help="test-time degradation (robustness evaluation)")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--no-samples", action="store_true", help="skip misclassified sample grid")
    return p


@torch.no_grad()
def run_inference(model, loader, device, task):
    model.eval()
    scores, preds, ya, yb = [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        out = model(imgs)
        if task == "binary":
            scores.append(torch.sigmoid(out).squeeze(-1).float().cpu().numpy())
            preds.append(None)
        elif task == "attribution":
            prob = torch.softmax(out.float(), dim=1)
            scores.append(prob[:, 1:].argmax(dim=1))  # placeholder, not used for attribution
            preds.append(prob.argmax(dim=1).cpu().numpy())
        else:  # multitask
            z_d, z_a = out
            scores.append(torch.sigmoid(z_d).float().cpu().numpy())
            preds.append(torch.softmax(z_a.float(), dim=1).argmax(dim=1).cpu().numpy())
        ya.append(batch["label_a"].numpy())
        yb.append(batch["label_b"].numpy())
    return scores, preds, np.concatenate(ya), np.concatenate(yb)


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model_for_eval(args.checkpoint, device=device)
    task = ckpt.get("task") or (ckpt.get("config") or {}).get("task") or "binary"
    report_dir = Path(args.report_dir or Path(args.checkpoint).parent / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    degradation = args.degradation
    ds = build_dataset(args.split, data_dir=args.data_dir, limit=args.limit,
                       norm=args.norm, img_size=args.img_size, degradation=degradation)
    loader = build_loader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, seed=0)
    scores, preds, ya, yb = run_inference(model, loader, device, task)

    result: dict = {
        "checkpoint": str(args.checkpoint),
        "task": task,
        "split": args.split,
        "degradation": degradation,
        "n": int(len(ya)),
    }
    det_scores = None
    mis_rows = []

    if task == "binary":
        det_scores = np.concatenate(scores)
        result["binary"] = binary_report(ya, det_scores)
        p_a = (det_scores >= 0.5).astype(np.int64)
        for i, (a, p) in enumerate(zip(ya, p_a)):
            if a != p:
                mis_rows.append({"index": i, "label_a": int(a), "pred_a": int(p),
                                 "score_a": float(det_scores[i])})
    elif task == "attribution":
        attr_pred = np.concatenate(preds)
        result["attribution"] = attribution_report(yb, attr_pred)
        rep.save_confusion_png(result["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix.png")
        rep.save_confusion_png(result["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix_norm.png", normalize=True)
        for i, (t, p) in enumerate(zip(yb, attr_pred)):
            if t != p:
                mis_rows.append({"index": i, "label_b": int(t), "pred_b": int(p),
                                 "score_b": None})
    else:  # multitask
        det_scores = np.concatenate(scores)
        attr_pred = np.concatenate(preds)
        result["binary"] = binary_report(ya, det_scores)
        result["attribution"] = attribution_report(yb, attr_pred)
        rep.save_confusion_png(result["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix.png")
        rep.save_confusion_png(result["attribution"]["confusion_matrix"], LABEL_B_NAMES,
                               report_dir / "confusion_matrix_norm.png", normalize=True)
        pa = (det_scores >= 0.5).astype(np.int64)
        for i, (a, p, t, q) in enumerate(zip(ya, pa, yb, attr_pred)):
            if a != p or t != q:
                mis_rows.append({"index": i, "label_a": int(a), "pred_a": int(p),
                                 "label_b": int(t), "pred_b": int(q),
                                 "score_a": float(det_scores[i])})

    if det_scores is not None:
        per_gen = per_generator_detection(ya, det_scores, yb)
        result["per_generator_detection"] = per_gen
        rep.save_per_generator_csv(per_gen, report_dir / "per_generator.csv")

    rep.save_misclassified_csv(mis_rows, report_dir / "misclassified.csv")
    rep.save_json(result, report_dir / "report.json")

    if not args.no_samples and hasattr(ds, "raw_image"):
        sample_idx = [r["index"] for r in mis_rows[:24]]
        pred_map = {r["index"]: {"true": r.get("label_a", r.get("label_b")),
                                 "pred": r.get("pred_a", r.get("pred_b"))}
                    for r in mis_rows}
        try:
            rep.save_samples_grid(ds, sample_idx, pred_map, report_dir / "misclassified_grid.png")
        except Exception as exc:  # noqa: BLE001 — grid is optional, never fatal
            print(f"[report] sample grid skipped: {exc}")

    print(json.dumps({k: v for k, v in result.items() if k != "per_generator_detection"},
                     indent=2, default=str))
    print(f"[report] artifacts -> {report_dir}")


if __name__ == "__main__":
    main()
