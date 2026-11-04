"""Training engine: one loop used by binary, attribution and multi-task models.

Single process (default) or torchrun/DDP multi-GPU: torchrun sets RANK /
WORLD_SIZE / LOCAL_RANK and this module detects them. Only rank 0 evaluates,
writes artifacts and uploads to Drive; other ranks wait at barriers.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from mscocoai.constants import IMAGENET_MEAN, IMAGENET_STD, SEED
from mscocoai.data import build_dataset, build_loader, compute_attr_class_weights, compute_pos_weight
from mscocoai.metrics import attribution_report, binary_report
from mscocoai.models import (
    EMA, CheckpointCorrupt, build_model, export_fp16, find_latest_valid_checkpoint,
    load_checkpoint, rotate_backups, save_checkpoint,
)
from mscocoai import drive_upload
from mscocoai.distributed import (
    all_reduce_sum, barrier, broadcast_bool, broadcast_float64,
    build_sampler, cleanup_distributed, init_distributed, is_main_process,
)
from mscocoai.results import ResultsLog


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def check_free_disk(path: str | os.PathLike, min_free_gb: float) -> tuple[bool, float]:
    import shutil
    usage = shutil.disk_usage(str(Path(path)))
    free_gb = usage.free / (1024 ** 3)
    return (free_gb >= min_free_gb, free_gb)


def _model_spec(cfg, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> dict[str, Any]:
    return {
        "task": cfg.task,
        "backbone": cfg.backbone,
        "freq_branch": getattr(cfg, "freq_branch", "none"),
        "freq_k": getattr(cfg, "freq_k", 32),
        "clip_init": getattr(cfg, "clip_init", "none"),
        "clip_unfreeze": getattr(cfg, "clip_unfreeze", "last2"),
        "img_size": getattr(cfg, "img_size", 224),
        "norm": getattr(cfg, "norm", "imagenet"),
        "mean": list(mean),
        "std": list(std),
    }


def _build_scheduler(optimizer, warmup_epochs: int, max_epochs: int) -> LambdaLR:
    warmup_epochs = max(1, int(warmup_epochs))
    max_epochs = max(warmup_epochs + 1, int(max_epochs))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _predict(model, loader, device, amp_device: str, task: str):
    """Per-rank predictions; the caller gathers across ranks when distributed."""
    model.eval()
    all_score, all_pred, all_a, all_b = [], [], [], []
    null_ctx = __import__("contextlib").nullcontext()
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        if amp_device == "cpu":
            ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        else:
            ctx = null_ctx
        with ctx:
            logits = model(imgs)
        if task == "binary":
            prob = torch.sigmoid(logits).float().squeeze(-1).cpu().numpy()
            all_score.append(prob)
            all_pred.append((prob >= 0.5).astype(np.int64))
        elif task == "attribution":
            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_score.append(prob)
            all_pred.append(prob.argmax(axis=1))
        elif task == "multitask":
            z_d, z_a = logits
            prob_d = torch.sigmoid(z_d).float().cpu().numpy()
            prob_a = torch.softmax(z_a.float(), dim=1).cpu().numpy()
            all_score.append(prob_d)
            all_pred.append((prob_d >= 0.5, prob_a.argmax(axis=1)))
        all_a.append(batch["label_a"].numpy())
        all_b.append(batch["label_b"].numpy())
    return (all_score, all_pred, np.concatenate(all_a), np.concatenate(all_b))


@torch.no_grad()
def _val_loss(model, loader, device, criterion_d, criterion_a, cfg, world_size: int = 1) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        la = batch["label_a"].to(device, non_blocking=True)
        lb = batch["label_b"].to(device, non_blocking=True)
        if cfg.task == "binary":
            z = model(imgs)
            if z.dim() > 1:
                z = z.squeeze(-1)
            loss = criterion_d(z, la.float())
        elif cfg.task == "attribution":
            z = model(imgs)
            loss = criterion_a(z, lb)
        else:
            z_d, z_a = model(imgs)
            loss = cfg.lambda_d * criterion_d(z_d, la.float()) + cfg.lambda_a * criterion_a(z_a, lb)
        total += float(loss) * imgs.size(0)
        n += imgs.size(0)
    if world_size > 1:
        total = all_reduce_sum(total)
        n = all_reduce_sum(n)
    return total / max(1, int(n))


def evaluate_split(model, loader, device, task: str, world_size: int = 1) -> dict:
    """Metrics for the split. Under DDP every rank evaluates the full split
    (identical weights), so ranks agree bitwise; rank 0 is the only writer."""
    amp_device = "cpu" if str(device) == "cpu" else ("cuda" if str(device).startswith("cuda") else "xla")
    all_score, all_pred, ya, yb = _predict(model, loader, device, amp_device, task)
    out: dict[str, Any] = {}
    if task == "binary":
        scores = np.concatenate(all_score)
        out.update(binary_report(ya, scores))
    elif task == "attribution":
        preds = np.concatenate(all_pred)
        out.update(attribution_report(yb, preds))
    else:
        det = np.concatenate([p[0] for p in all_pred])
        att = np.concatenate([p[1] for p in all_pred])
        out["detection"] = binary_report(ya, det)
        out["attribution"] = attribution_report(yb, att)
    return out


def _binary_val_scores(model, loader, device, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, scores) for the detection head — used by --calibrate only."""
    amp_device = "cpu" if str(device) == "cpu" else ("cuda" if str(device).startswith("cuda") else "xla")
    all_score, _, ya, _ = _predict(model, loader, device, amp_device, task)
    return ya, np.concatenate(all_score)


def _save_all(model, optimizer, scheduler, scaler, ema, cfg, epoch, best_val, out_dir: Path,
              is_best: bool, val_metrics: dict, step_in_epoch: int = 0,
              sampler_epoch: int = 0, threshold: float | None = None) -> None:
    state = {
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "sampler_epoch": int(sampler_epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": (ema.shadow if ema is not None else None),
        "config": vars(cfg) if not isinstance(cfg, dict) else cfg,
        "task": cfg.task,
        "backbone": cfg.backbone,
        "best_val": float(best_val),
        "val_metrics": val_metrics,
        "model_spec": _model_spec(cfg),
        "threshold": threshold,
    }
    rotate_backups(out_dir, "last.pt")
    save_checkpoint(state, out_dir / "last.pt")
    (out_dir / "LATEST").write_text("last.pt")
    if is_best:
        save_checkpoint(state, out_dir / "best.pt")
    export_fp16(out_dir / "last.pt", out_dir / "last_weights.pt")


def _maybe_upload(out_dir: Path, cfg) -> None:
    backend = getattr(cfg, "drive_backend", "none")
    if backend in (None, "none"):
        return
    files = [out_dir / "last_weights.pt"]
    if getattr(cfg, "upload_metrics", True):
        files += [out_dir / "history.jsonl", out_dir / "metrics.csv", out_dir / "results.json"]
    files = [f for f in files if f.is_file()]
    if not files:
        return
    run_name = cfg.out_dir.name if isinstance(cfg.out_dir, Path) else Path(cfg.out_dir).name
    try:
        drive_upload.upload_files(files, backend=backend, run_name=run_name,
                                  drive_dir=getattr(cfg, "drive_dir", None),
                                  upload_url=getattr(cfg, "apps_script_url", None),
                                  token=getattr(cfg, "apps_script_token", None))
    except Exception as exc:  # noqa: BLE001 — never crash training because of Drive
        print(f"[drive_upload] WARNING: upload failed ({exc}); training continues.")


def _flush_accum(model, optimizer, scaler, use_amp: bool, ema) -> None:
    if use_amp:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(model)


def _prior_best_epoch(log: ResultsLog) -> int | None:
    best_epoch, best_val = None, float("inf")
    for row in log.epoch_rows():
        v = row.get("val_loss")
        if v is not None and float(v) < best_val:
            best_val = float(v)
            best_epoch = int(row.get("epoch", 0))
    return best_epoch


def _resave_with_threshold(checkpoint: Path, threshold: float) -> None:
    from mscocoai.models import atomic_torch_save
    ckpt = load_checkpoint(checkpoint)
    ckpt["threshold"] = float(threshold)
    atomic_torch_save(ckpt, checkpoint)


def fit(cfg) -> dict:
    seed_everything(cfg.seed)

    # ---- decide device + backend from --device and the environment (before group init)
    requested = (getattr(cfg, "device", "auto") or "auto").lower()
    env_ws = int(os.environ.get("WORLD_SIZE", "1"))
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    allow_gpu_share = os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE", "").lower() in ("1", "true", "yes")

    if requested == "cpu":
        device, backend = "cpu", "gloo"
    elif requested == "tpu":
        device, backend = "tpu", "gloo"
    elif requested.startswith("cuda"):
        if env_ws > 1 and n_gpu and env_ws > n_gpu and not allow_gpu_share:
            print(f"[ddp] {env_ws} ranks but only {n_gpu} GPU(s); falling back to CPU+gloo. "
                  f"Set NCCL_MULTI_RANK_GPU_ENABLE=1 to share a GPU (or use fewer ranks).")
            device, backend = "cpu", "gloo"
        else:
            device, backend = requested, "nccl"
    elif requested == "auto":
        if n_gpu > 0 and env_ws <= n_gpu:
            device, backend = "cuda", "nccl"
        else:
            # piling more ranks than GPUs onto one GPU crashes nccl; fall back instead
            device, backend = "cpu", "gloo"
    else:
        raise ValueError(f"unknown --device: {requested}")

    rank, world_size, local_rank = init_distributed(backend=backend)
    is_main = rank == 0

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- preflight
    ok, free_gb = check_free_disk(out_dir, getattr(cfg, "min_free_gb", 5.0))
    if not ok:
        raise SystemExit(
            f"Not enough free disk on {out_dir} ({free_gb:.1f} GB available, "
            f"need ≥ {getattr(cfg, 'min_free_gb', 5.0)} GB). "
            "Free space or lower --min-free-gb.")

    tpu_mode = device == "tpu"
    if tpu_mode:
        from mscocoai import xla
        if not xla.xla_available():
            raise SystemExit("--device tpu requested but torch_xla is not installed/available.")
        torch_device = xla.xla_device()
        xla.xla_seed(cfg.seed)
        amp_device = "xla"
        autocast_ctx = xla.xla_amp_context()
    elif device == "cuda" or device.startswith("cuda:"):
        torch_device = device if world_size == 1 else f"cuda:{local_rank}"
        amp_device = "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=True)
    else:
        torch_device = "cpu"
        amp_device = "cpu"
        autocast_ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)

    out_cfg = vars(cfg) if not isinstance(cfg, dict) else dict(cfg)
    log = ResultsLog(out_dir, run_name=out_dir.name)
    best_epoch = _prior_best_epoch(log)
    stop_reason = "completed"
    t_start = time.time()

    # ---------------- data
    holdout_gen = getattr(cfg, "holdout_generator", None)
    data_layout = getattr(cfg, "data_layout", "auto")
    train_ds = build_dataset("train", data_dir=cfg.data_dir, limit=cfg.limit,
                             norm=cfg.norm, img_size=cfg.img_size,
                             layout=data_layout, holdout_generator=holdout_gen)
    val_ds = build_dataset("validation", data_dir=cfg.data_dir, limit=cfg.limit,
                           norm=cfg.norm, img_size=cfg.img_size,
                           layout=data_layout, holdout_generator=holdout_gen)
    num_workers = cfg.num_workers
    train_sampler = build_sampler(train_ds, shuffle=True, seed=cfg.seed, drop_last=True) \
        if world_size > 1 else None
    train_loader = build_loader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=num_workers, seed=cfg.seed,
                                balanced_sampler=cfg.balance_sampler, drop_last=True,
                                sampler=train_sampler)
    val_loader = build_loader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
                              num_workers=num_workers, seed=cfg.seed)

    # ---------------- model & losses
    model = build_model(cfg.task, cfg.backbone, pretrained=True,
                        freq_branch=getattr(cfg, "freq_branch", "none"),
                        freq_k=getattr(cfg, "freq_k", 32),
                        clip_init=getattr(cfg, "clip_init", "none"),
                        clip_unfreeze=getattr(cfg, "clip_unfreeze", "last2"),
                        img_size=getattr(cfg, "img_size", 224))
    if getattr(cfg, "channels_last", False) and amp_device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(torch_device)
    raw_model = model  # un-wrapped reference for saving/eval
    if world_size > 1:
        device_ids = [local_rank] if amp_device == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=device_ids)

    norm_labels = getattr(train_ds, "labels_b", lambda: np.array([]))()
    if cfg.task == "attribution" or cfg.task == "multitask":
        attr_weights = None
        if cfg.attr_class_weights:
            attr_weights = torch.tensor(compute_attr_class_weights(norm_labels),
                                        dtype=torch.float32, device=torch_device)
        criterion_a = nn.CrossEntropyLoss(weight=attr_weights,
                                          label_smoothing=cfg.label_smoothing)
    else:
        criterion_a = None
    if cfg.task == "binary" or cfg.task == "multitask":
        if cfg.balance_loss and hasattr(train_ds, "labels_b"):
            pos_weight = compute_pos_weight((train_ds.labels_b() != 0).astype(np.int64))
            pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=torch_device)
        elif cfg.balance_loss:
            pos_weight_t = torch.tensor([1.0], dtype=torch.float32, device=torch_device)
        else:
            pos_weight_t = None
        criterion_d = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        criterion_d = None

    optimizer = torch.optim.AdamW([p for p in raw_model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.GradScaler(enabled=amp_device == "cuda")
    ema = EMA(raw_model, decay=cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None

    # ---------------- resume
    start_epoch, best_val, patience = 0, float("inf"), 0
    threshold: float | None = None
    skip_batches = 0
    if cfg.resume and cfg.resume != "none":
        ckpt_path = None
        if cfg.resume == "auto":
            ckpt_path = find_latest_valid_checkpoint(out_dir)
        elif cfg.resume == "weights-only":
            cand = out_dir / "last_weights.pt"
            ckpt_path = cand if cand.is_file() else None
        else:
            ckpt_path = Path(cfg.resume)
        if ckpt_path is not None:
            try:
                ckpt = load_checkpoint(ckpt_path, map_location=torch_device)
                raw_model.load_state_dict(ckpt["model"])
                step_in_epoch = int(ckpt.get("step_in_epoch", 0) or 0)
                if getattr(cfg, "resume_epoch_only", False):
                    step_in_epoch = 0
                if step_in_epoch:
                    start_epoch = int(ckpt.get("epoch", -1))
                    skip_batches = step_in_epoch
                else:
                    start_epoch = int(ckpt.get("epoch", -1)) + 1
                threshold = ckpt.get("threshold")
                if cfg.resume != "weights-only" and ckpt.get("optimizer"):
                    optimizer.load_state_dict(ckpt["optimizer"])
                    if ckpt.get("scheduler"):
                        scheduler.load_state_dict(ckpt["scheduler"])
                    if ckpt.get("scaler") and amp_device == "cuda":
                        scaler.load_state_dict(ckpt["scaler"])
                    if ema is not None and ckpt.get("ema"):
                        ema.shadow = {k: v.to(torch_device) for k, v in ckpt["ema"].items()}
                    best_val = float(ckpt.get("best_val", float("inf")))
                    print(f"[resume] {ckpt_path} -> start_epoch={start_epoch} "
                          f"resume_step={skip_batches} best_val={best_val:.4f}")
                else:
                    print(f"[resume] weights-only from {ckpt_path}; optimizer/scheduler reset")
            except CheckpointCorrupt as exc:
                raise SystemExit(f"Resume failed: {exc}") from exc
        else:
            print("[resume] no valid checkpoint found; starting from scratch")

    # ---------------- training loop
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        micro = 0
        budget_hit = False
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            if skip_batches > 0:
                skip_batches -= 1
                continue
            imgs = batch["image"].to(torch_device, non_blocking=True)
            la = batch["label_a"].to(torch_device, non_blocking=True)
            lb = batch["label_b"].to(torch_device, non_blocking=True)
            with autocast_ctx:
                if cfg.task == "binary":
                    _z = model(imgs)
                    loss = criterion_d(_z.squeeze(-1) if _z.dim() > 1 else _z, la.float())
                elif cfg.task == "attribution":
                    loss = criterion_a(model(imgs), lb)
                else:
                    z_d, z_a = model(imgs)
                    loss = cfg.lambda_d * criterion_d(z_d, la.float()) + cfg.lambda_a * criterion_a(z_a, lb)
                loss = loss / cfg.accum_steps
            if amp_device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro += 1
            running += float(loss.detach()) * cfg.accum_steps * imgs.size(0)
            seen += imgs.size(0)
            if micro % cfg.accum_steps == 0:
                if amp_device == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if amp_device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(raw_model)
                if tpu_mode:
                    from mscocoai import xla
                    xla.xla_mark_step()
                global_step = (epoch * len(train_loader)) + micro
                if cfg.save_steps and cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    if is_main:
                        _save_all(raw_model, optimizer, scheduler, scaler, ema, cfg, epoch,
                                  best_val, out_dir, is_best=False, val_metrics={},
                                  step_in_epoch=micro, sampler_epoch=epoch,
                                  threshold=threshold)
                        log.append({"type": "partial", "epoch": epoch, "step": micro,
                                    "train_loss": running / max(1, seen)})
                        log.flush({**out_cfg, "threshold": threshold, "stop_reason": stop_reason})
                        _maybe_upload(out_dir, cfg)
            if cfg.max_hours:
                if world_size > 1:
                    hit = broadcast_bool(is_main and (time.time() - t_start) > cfg.max_hours * 3600)
                else:
                    hit = (time.time() - t_start) > cfg.max_hours * 3600
                if hit:
                    budget_hit = True
                    break
        if micro % cfg.accum_steps != 0:
            _flush_accum(model, optimizer, scaler, amp_device == "cuda", ema)

        if budget_hit:
            stop_reason = "time_budget"
            elapsed = running / max(1, seen)
            if is_main:
                log.append({"type": "partial", "epoch": epoch, "step": micro,
                            "train_loss": elapsed})
                _save_all(raw_model, optimizer, scheduler, scaler, ema, cfg, epoch,
                          best_val, out_dir, is_best=False, val_metrics={},
                          step_in_epoch=micro, sampler_epoch=epoch,
                          threshold=threshold)
                _maybe_upload(out_dir, cfg)
            break
        scheduler.step()

        # ---------------- validation
        eval_model = raw_model
        backup = None
        if ema is not None:
            backup = ema.store(eval_model)
            ema.copy_to(eval_model)
        val_loss = _val_loss(eval_model, val_loader, torch_device, criterion_d, criterion_a, cfg,
                             world_size)
        if world_size > 1:
            val_loss = broadcast_float64(val_loss, src=0)
        val_metrics = evaluate_split(eval_model, val_loader, torch_device, cfg.task, world_size) \
            if cfg.val_metrics_each_epoch else {"val_loss": val_loss}
        if ema is not None:
            ema.restore(eval_model, backup)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if is_main:
            _save_all(raw_model, optimizer, scheduler, scaler, ema, cfg, epoch, best_val,
                      out_dir, is_best=is_best, val_metrics=val_metrics,
                      step_in_epoch=0, sampler_epoch=epoch, threshold=threshold)
            row = {"type": "epoch", "epoch": epoch, "train_loss": running / max(1, seen),
                   "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "is_best": is_best,
                   "elapsed_sec": round(time.time() - t0, 1), "val_metrics": val_metrics}
            log.append(row)
            log.flush({**out_cfg, "threshold": threshold, "stop_reason": stop_reason})
            _maybe_upload(out_dir, cfg)
            print(f"epoch {epoch}: train={row['train_loss']:.4f} val={val_loss:.4f} "
                  f"lr={row['lr']:.2e}{' *best*' if is_best else ''} ({row['elapsed_sec']}s)")
        barrier()
        if patience >= cfg.early_patience:
            if is_main:
                print(f"early stopping: no val improvement for {patience} epochs")
            break

    # ---------------- calibration (validation split only)
    if is_main and getattr(cfg, "calibrate", "off") != "off" and cfg.task in ("binary", "multitask"):
        from mscocoai.calibrate import fit_threshold
        ya, scores = _binary_val_scores(raw_model, val_loader, torch_device, cfg.task)
        try:
            threshold = fit_threshold(ya, scores, objective=cfg.calibrate)
            _resave_with_threshold(out_dir / "last.pt", threshold)
            print(f"[calibrate] threshold={threshold:.4f} (objective={cfg.calibrate})")
        except Exception as exc:  # noqa: BLE001 — calibration must never crash training
            print(f"[calibrate] WARNING: threshold fitting failed ({exc}); keeping 0.5")

    # ---------------- final flush
    if is_main:
        log.flush({**out_cfg, "threshold": threshold, "stop_reason": stop_reason})
    barrier()
    cleanup_distributed()
    return {"best_val": best_val, "best_epoch": best_epoch, "out_dir": str(out_dir),
            "stop_reason": stop_reason, "history_path": str(log.history_path)}
