"""CLIP (ViT-B/16) OpenAI backbone + lightweight head — the UniFD-style branch.

Frozen paper protocol (default):
  python -m mscocoai.clip_probe --mode extract --out-dir outputs/clip_vitb16
  python -m mscocoai.clip_probe --mode train   --task binary --out-dir outputs/clip_vitb16
  python -m mscocoai.clip_probe --mode eval    --task binary --out-dir outputs/clip_vitb16

SOTA levers (optional, changes the backbone — disclose when reporting):
  --train-mode last2   unfreeze the last 2 transformer blocks + LN + projection
  --train-mode full    full end-to-end fine-tune (use --unfreeze-lr, default 1e-5)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mscocoai.constants import NUM_ATTR_CLASSES, SEED
from mscocoai.data import (
    build_dataset, build_loader, compute_attr_class_weights, compute_pos_weight,
)
from mscocoai.metrics import attribution_report, binary_report, per_generator_detection
from mscocoai.models import atomic_torch_save, load_checkpoint
from mscocoai import report as rep
from mscocoai.engine import seed_everything


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLIP embedding extraction and head training")
    p.add_argument("--mode", required=True, choices=["extract", "train", "eval"])
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--task", default="binary", choices=["binary", "attribution"])
    p.add_argument("--train-mode", default="frozen", choices=["frozen", "last2", "full"])
    p.add_argument("--unfreeze-lr", type=float, default=1e-5, help="lr for last2/full fine-tuning")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=15, help="frozen probe; ignored for fine-tune (uses --ft-epochs)")
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3, help="frozen head lr (paper: 1e-3)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-patience", type=int, default=3)
    p.add_argument("--attr-class-weights", action="store_true")
    p.add_argument("--balance-loss", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--report-dir", default=None)
    return p


def _cache_paths(out_dir: Path, split: str) -> tuple[Path, Path, Path]:
    return (out_dir / f"embeddings_{split}.npy",
            out_dir / f"label_a_{split}.npy",
            out_dir / f"label_b_{split}.npy")


def _load_cache(out_dir: Path, split: str):
    emb_path, la_path, lb_path = _cache_paths(out_dir, split)
    if not emb_path.exists():
        raise SystemExit(f"Missing embedding cache: {emb_path}. Run --mode extract first.")
    return np.load(emb_path), np.load(la_path), np.load(lb_path)


def _create_clip(args, device):
    import open_clip  # lazy import
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)
    if args.train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    elif args.train_mode == "last2":
        for name, p in model.named_parameters():
            p.requires_grad_(any(s in name for s in ("resblocks.10", "resblocks.11", "ln_post", "proj")))
        model.train()
    else:  # full
        model.train()
    return model


@torch.no_grad()
def _extract_split(model, ds, device, out_dir: Path, split: str, batch_size: int, num_workers: int) -> None:
    emb_path, la_path, lb_path = _cache_paths(out_dir, split)
    if emb_path.exists() and la_path.exists() and lb_path.exists():
        print(f"[extract] {split}: cache hit -> {emb_path.name}")
        return
    loader = build_loader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, seed=0)
    feats, la_all, lb_all = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            f = model.encode_image(imgs)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats.append(f.cpu().numpy())
        la_all.append(batch["label_a"].numpy()); lb_all.append(batch["label_b"].numpy())
    np.save(emb_path, np.concatenate(feats).astype(np.float32))
    np.save(la_path, np.concatenate(la_all))
    np.save(lb_path, np.concatenate(lb_all))
    print(f"[extract] {split}: {emb_path.stat().st_size / 1e6:.1f} MB -> {emb_path}")


def _train_frozen_probe(args, out_dir: Path, device: str) -> None:
    Xtr, yatr, ybtr = _load_cache(out_dir, "train")
    Xva, yava, ybva = _load_cache(out_dir, "validation")
    seed_everything(args.seed)
    dim = Xtr.shape[1]
    out_dim = 1 if args.task == "binary" else NUM_ATTR_CLASSES
    head = nn.Linear(dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.task == "binary":
        if args.balance_loss:
            pw = compute_pos_weight(yatr)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()
        ytr = torch.tensor(yatr, dtype=torch.float32, device=device)
        yva = torch.tensor(yava, dtype=torch.float32, device=device)
    else:
        w = None
        if args.attr_class_weights:
            w = torch.tensor(compute_attr_class_weights(ybtr), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
        ytr = torch.tensor(ybtr, dtype=torch.long, device=device)
        yva = torch.tensor(ybva, dtype=torch.long, device=device)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    def loss_for(X, y):
        z = head(X)
        return criterion(z.squeeze(-1) if args.task == "binary" else z, y)

    best_val, patience, best_state = float("inf"), 0, None
    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        tr_loss = loss_for(Xtr_t, ytr)
        tr_loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            va_loss = float(loss_for(Xva_t, yva))
        if va_loss < best_val:
            best_val, patience = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
        print(f"probe epoch {epoch}: train={tr_loss.detach().item():.4f} val={va_loss:.4f}{' *best*' if patience == 0 else ''}")
        if patience >= args.early_patience:
            print("probe early stopping")
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    ckpt = {"model": head.state_dict(), "config": vars(args),
            "task": f"clip_{args.task}", "backbone": f"clip_{args.model}",
            "val_metrics": {"val_loss": float(best_val)}}
    atomic_torch_save(ckpt, out_dir / f"probe_{args.task}.pt")
    print(f"[probe] saved -> {out_dir / f'probe_{args.task}.pt'}")


def _train_unfrozen(args, out_dir: Path, device: str) -> None:
    """End-to-end fine-tune of the visual encoder (+ head) — SOTA lever, disclose it."""
    seed_everything(args.seed)
    train_ds = build_dataset("train", data_dir=args.data_dir, limit=args.limit, norm="clip", img_size=args.img_size)
    val_ds = build_dataset("validation", data_dir=args.data_dir, limit=args.limit, norm="clip", img_size=args.img_size)
    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, seed=args.seed, drop_last=True)
    val_loader = build_loader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, seed=args.seed)
    model = _create_clip(args, device)
    out_dim = 1 if args.task == "binary" else NUM_ATTR_CLASSES
    with torch.no_grad():
        dummy = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
        feat_dim = model.encode_image(dummy).shape[1]
    head = nn.Linear(feat_dim, out_dim).to(device)
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.unfreeze_lr},
              {"params": head.parameters(), "lr": args.unfreeze_lr}]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    if args.task == "binary":
        if args.balance_loss:
            pw = compute_pos_weight((train_ds.labels_b() != 0).astype(np.int64))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        w = None
        if args.attr_class_weights:
            w = torch.tensor(compute_attr_class_weights(train_ds.labels_b()),
                             dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)

    class _Wrapped(nn.Module):
        def __init__(self, enc, head, task):
            super().__init__(); self.enc, self.head, self.task = enc, head, task
        def forward(self, x):
            f = self.enc.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            z = self.head(f)
            return z.squeeze(-1) if self.task == "binary" else z

    wrapped = _Wrapped(model, head, args.task).to(device)
    scaler = torch.GradScaler(enabled=device == "cuda")
    best_val, patience = float("inf"), 0
    hist = out_dir / "ft_history.jsonl"
    for epoch in range(args.ft_epochs):
        t0 = time.time()
        model.train(); head.train(); wrapped.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16,
                                enabled=True):
                z = wrapped(imgs)
                if args.task == "binary":
                    loss = criterion(z, batch["label_a"].to(device, non_blocking=True).float())
                else:
                    loss = criterion(z, batch["label_b"].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running += float(loss) * imgs.size(0); seen += imgs.size(0)
        # validation loss
        model.eval(); head.eval(); wrapped.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                z = wrapped(imgs)
                if args.task == "binary":
                    loss = criterion(z, batch["label_a"].to(device).float())
                else:
                    loss = criterion(z, batch["label_b"].to(device))
                tot += float(loss) * imgs.size(0); n += imgs.size(0)
        va_loss = tot / max(1, n)
        is_best = va_loss < best_val
        best_val = va_loss if is_best else best_val
        patience = 0 if is_best else patience + 1
        ckpt = {"model": wrapped.state_dict(), "config": vars(args),
                "task": f"clip_{args.task}", "backbone": f"clip_{args.model}",
                "val_metrics": {"val_loss": float(va_loss)}}
        if is_best:
            atomic_torch_save(ckpt, out_dir / f"clip_ft_{args.task}_best.pt")
        atomic_torch_save(ckpt, out_dir / f"clip_ft_{args.task}_last.pt")
        (out_dir / "LATEST").write_text(f"clip_ft_{args.task}_last.pt")
        print(f"ft epoch {epoch}: train={running / max(1, seen):.4f} val={va_loss:.4f}{' *best*' if is_best else ''} ({time.time() - t0:.0f}s)")
        if patience >= args.early_patience:
            print("fine-tune early stopping"); break


def _eval(args, out_dir: Path, device: str) -> None:
    split = "test"
    report_dir = Path(args.report_dir or out_dir / f"report_{args.task}")
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.train_mode == "frozen":
        Xte, yate, ybte = _load_cache(out_dir, split)
        ckpt = load_checkpoint(out_dir / f"probe_{args.task}.pt", map_location=device)
        dim = Xte.shape[1]
        head = nn.Linear(dim, 1 if args.task == "binary" else NUM_ATTR_CLASSES).to(device)
        head.load_state_dict(ckpt["model"]); head.eval()
        with torch.no_grad():
            z = head(torch.tensor(Xte, dtype=torch.float32, device=device))
        result = _scores_to_reports(args.task, z, yate, ybte, report_dir)
    else:
        ckpt = load_checkpoint(out_dir / f"clip_ft_{args.task}_best.pt", map_location=device)
        import open_clip
        enc, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        enc = enc.to(device)
        feat_dim = enc.encode_image(torch.zeros(1, 3, args.img_size, args.img_size, device=device)).shape[1]
        head = nn.Linear(feat_dim, 1 if args.task == "binary" else NUM_ATTR_CLASSES).to(device)
        from collections import OrderedDict
        enc_sd = OrderedDict((k[4:] if k.startswith("enc.") else k, v) for k, v in ckpt["model"].items()
                             if not k.startswith("head."))
        head_sd = {k[5:]: v for k, v in ckpt["model"].items() if k.startswith("head.")}
        enc.load_state_dict(enc_sd, strict=False)
        head.load_state_dict(head_sd)
        enc.eval(); head.eval()
        ds = build_dataset(split, data_dir=args.data_dir, limit=args.limit, norm="clip", img_size=args.img_size)
        loader = build_loader(ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, seed=0)
        zs, yate, ybte = [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True)
                f = enc.encode_image(imgs).float()
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                zs.append(head(f))
                yate.append(batch["label_a"].numpy()); ybte.append(batch["label_b"].numpy())
        z = torch.cat(zs)
        result = _scores_to_reports(args.task, z, np.concatenate(yate), np.concatenate(ybte), report_dir)
    rep.save_json(result, report_dir / "report.json")
    import json
    print(json.dumps(result, indent=2, default=str))
    print(f"[report] -> {report_dir}")


def _scores_to_reports(task, z, yate, ybte, report_dir: Path) -> dict:
    if task == "binary":
        scores = torch.sigmoid(z.squeeze(-1)).cpu().numpy() if torch.is_tensor(z) else z
        result = {"binary": binary_report(yate, scores),
                  "per_generator_detection": per_generator_detection(yate, scores, ybte)}
        rep.save_per_generator_csv(result["per_generator_detection"], report_dir / "per_generator.csv")
    else:
        pred = z.argmax(dim=1).cpu().numpy() if torch.is_tensor(z) else z.argmax(axis=1)
        result = {"attribution": attribution_report(ybte, pred)}
        rep.save_confusion_png(result["attribution"]["confusion_matrix"],
                               list(result["attribution"]["labels"]),
                               report_dir / "confusion_matrix.png")
    result["split"] = "test"
    return result


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "extract":
        model = _create_clip(args, device)
        for split in ("train", "validation", "test"):
            ds = build_dataset(split, data_dir=args.data_dir, limit=args.limit,
                               norm="clip", img_size=args.img_size)
            _extract_split(model, ds, device, out_dir, split, args.batch_size, args.num_workers)
        return
    if args.mode == "train":
        if args.train_mode == "frozen":
            _train_frozen_probe(args, out_dir, device)
        else:
            _train_unfrozen(args, out_dir, device)
        return
    if args.mode == "eval":
        _eval(args, out_dir, device)
        print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
