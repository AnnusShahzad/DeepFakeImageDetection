"""Data loading for MS-COCOAI (HF hub or a local folder mirror).

Local mirror layout (produced by rsync/wget or manual download):

    <root>/train/{real,sd21,sdxl,sd3,dalle3,mj6}/*.{jpg,jpeg,png,webp}
    <root>/validation/...
    <root>/test/...

HF hub source (default): https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from mscocoai.constants import (
    CLIP_MEAN, CLIP_STD, HF_DATASET_ID, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    LABEL_B_NAMES, SEED, SPLIT_ALIASES,
)

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


_HOLDOUT_IDS = {name.lower(): i for i, name in enumerate(LABEL_B_NAMES)}


def _holdout_id(name: str | None) -> int | None:
    """Map a holdout generator name to its Label_B index (generators only)."""
    if not name:
        return None
    key = str(name).lower()
    if key == "real":
        raise ValueError("holdout_generator must be one of the five generators "
                         "(sd21, sdxl, sd3, dalle3, mj6), not 'real'")
    try:
        return _HOLDOUT_IDS[key]
    except KeyError:
        raise ValueError(f"unknown holdout generator: {name!r}; expected one of "
                         f"{[n for n in _HOLDOUT_IDS if n != 'real']}") from None


def detect_layout(root: str | os.PathLike) -> str:
    """Return 'parquet', 'folder' or 'unknown' for a dataset root directory."""
    root = Path(root)
    if not root.is_dir():
        return "unknown"
    for candidate in (root, root / "data"):
        if candidate.is_dir() and any(candidate.glob("*.parquet")):
            return "parquet"
    for s in ("train", "validation", "test"):
        if (root / s).is_dir():
            return "folder"
    return "unknown"


def canonical_split(split: str) -> str:
    split = split.lower()
    return SPLIT_ALIASES.get(split, split)


def _norm_pair(norm: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if norm == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if norm == "clip":
        return CLIP_MEAN, CLIP_STD
    raise ValueError(f"unknown norm: {norm}")


class RandomJPEGCompression:
    """Re-encode through JPEG with a random quality — robustness to compression artifacts."""

    def __init__(self, quality: tuple[int, int] = (50, 95), p: float = 0.5):
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img.convert("RGB")
        q = random.randint(int(self.quality[0]), int(self.quality[1]))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 5, sigma: tuple[float, float] = (0.1, 2.0), p: float = 0.25):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        r = random.uniform(*self.sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def _jpeg_q75(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur_15(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


DEGRADATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "none": lambda img: img,
    "jpeg75": _jpeg_q75,
    "blur": _blur_15,
}


def get_transforms(kind: str, norm: str = "imagenet", img_size: int = IMG_SIZE,
                   degradation: str = "none") -> Callable:
    """kind: train | val | test (val == test, deterministic)."""
    mean, std = _norm_pair(norm)
    degrade = DEGRADATIONS[degradation]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if kind == "train":
        ops: list[Callable] = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            RandomGaussianBlur(p=0.25),
            RandomJPEGCompression(quality=(50, 95), p=0.5),
            to_tensor,
        ]
        return transforms.Compose(ops)
    if kind in ("val", "test"):
        return transforms.Compose([
            transforms.Lambda(degrade),
            transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            to_tensor,
        ])
    raise ValueError(f"unknown transform kind: {kind}")


class LocalFolderDataset(Dataset):
    """Reads from a local mirror: root/<split>/<label_name>/<image>."""

    def __init__(self, root: str | os.PathLike, split: str, transform: Callable | None = None,
                 limit: int | None = None, holdout_generator: str | None = None):
        self.root = Path(root)
        self.split = canonical_split(split)
        self.transform = transform
        self.holdout_generator = holdout_generator
        self.samples: list[tuple[Path, int]] = []
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Local split directory not found: {split_dir}\n"
                "Expected <root>/{train,validation,test}/{real,sd21,sdxl,sd3,dalle3,mj6}/.\n"
                "See README.md §Dataset or pass --data-dir to a folder that has this layout."
            )
        for label_id, name in enumerate(LABEL_B_NAMES):
            cls_dir = split_dir / name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMG_EXT:
                    self.samples.append((p, label_id))
        holdout_id = _holdout_id(holdout_generator)
        if holdout_id is not None:
            self.samples = [s for s in self.samples if s[1] != holdout_id]
        if limit:
            labels = [s[1] for s in self.samples]
            idx = stratified_limit_indices(labels, limit, seed=SEED)
            self.samples = [self.samples[i] for i in idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path, label_b = self.samples[index]
        img = Image.open(path).convert("RGB")
        label_a = 0 if label_b == 0 else 1
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label_a": label_a, "label_b": label_b, "index": index}

    def labels_b(self) -> np.ndarray:
        return np.array([s[1] for s in self.samples], dtype=np.int64)

    def raw_image(self, index: int):
        path, _ = self.samples[int(index)]
        return Image.open(path).convert("RGB")


class HfImageDataset(Dataset):
    """Wraps the HF hub dataset; images are decoded lazily per __getitem__."""

    def __init__(self, split: str, transform: Callable | None = None,
                 limit: int | None = None, cache_dir: str | None = None,
                 holdout_generator: str | None = None):
        from datasets import load_dataset  # lazy import: not needed for local mirrors/tests
        self.split = canonical_split(split)
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {self.split}")
        self.hf = load_dataset(HF_DATASET_ID, split=self.split, cache_dir=cache_dir)
        holdout_id = _holdout_id(holdout_generator)
        if holdout_id is not None:
            self.hf = self.hf.filter(lambda row: int(row["Label_B"]) != holdout_id,
                                      desc=f"holdout {holdout_generator}")
        if limit:
            labels = self.hf["Label_B"]
            idx = stratified_limit_indices(labels, limit, seed=SEED)
            self.hf = self.hf.select([int(i) for i in idx])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, index: int) -> dict:
        row = self.hf[index]
        img = row["Image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]) if isinstance(img, dict) else img)
        img = img.convert("RGB")
        label_a = int(row["Label_A"])
        label_b = int(row["Label_B"])
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label_a": label_a, "label_b": label_b, "index": index}

    def labels_b(self) -> np.ndarray:
        return np.array(self.hf["Label_B"], dtype=np.int64)

    def raw_image(self, index: int):
        row = self.hf[int(index)]
        img = row["Image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]) if isinstance(img, dict) else img)
        return img.convert("RGB")


def build_dataset(split: str, *, data_dir: str | None, limit: int | None,
                  norm: str, img_size: int, degradation: str = "none",
                  layout: str = "auto", holdout_generator: str | None = None) -> Dataset:
    split = canonical_split(split)
    kind = "train" if split == "train" else "test"
    tf = get_transforms(kind, norm=norm, img_size=img_size, degradation=degradation)
    if data_dir:
        root = Path(data_dir)
        resolved = layout
        if resolved == "auto":
            resolved = detect_layout(root)
        if resolved == "parquet":
            return ParquetImageDataset(root, split, transform=tf, limit=limit,
                                        holdout_generator=holdout_generator)
        if resolved == "folder":
            return LocalFolderDataset(root, split, transform=tf, limit=limit,
                                      holdout_generator=holdout_generator)
        raise FileNotFoundError(
            f"--data-dir {data_dir} does not contain a recognised dataset layout.\n"
            "Expected either <root>/{train,validation,test}/<real|sd21|sdxl|sd3|dalle3|mj6>/ "
            "(folder mirror) or <root>/data/<split>-*.parquet (HF parquet dump).")
    return HfImageDataset(split, transform=tf, limit=limit,
                          cache_dir=os.environ.get("HF_DATASETS_CACHE") or None,
                          holdout_generator=holdout_generator)


def build_loader(ds: Dataset, *, batch_size: int, shuffle: bool, num_workers: int = 4,
                 seed: int = SEED, balanced_sampler: bool = False,
                 drop_last: bool = False, sampler=None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if num_workers == "auto":
        num_workers = min(8, max(2, (os.cpu_count() or 2) // 2))
        if (os.cpu_count() or 2) <= 2:
            num_workers = 0
    elif isinstance(num_workers, str):
        try:
            num_workers = int(num_workers)
        except ValueError:
            num_workers = 4
    kwargs = dict(dataset=ds, batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), collate_fn=collate_dict,
                  generator=generator, drop_last=drop_last)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    if sampler is not None:
        if balanced_sampler:
            raise ValueError("sampler and balanced_sampler are mutually exclusive")
        kwargs.update(sampler=sampler, shuffle=False)
    elif balanced_sampler:
        labels = ds.labels_b()
        counts = np.bincount(labels, minlength=len(LABEL_B_NAMES))
        weights = (1.0 / np.maximum(counts, 1))[labels]
        sampler = WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(labels), replacement=True,
                                        generator=generator)
        kwargs.update(sampler=sampler, shuffle=False)
    else:
        kwargs.update(shuffle=shuffle)
    return DataLoader(**kwargs)


def stratified_limit_indices(labels, limit: int, seed: int = SEED) -> np.ndarray:
    """Deterministic class-balanced subsample indices for `--limit`.

    A naive `samples[:limit]` would take the first N files in class order
    (i.e. all real + sd21 + sdxl + a prefix of sd3) and silently exclude
    dalle3/mj6 — making any smoke/debug subset useless. This picks an equal
    number per class (remainder distributed to the first classes), seeded.
    """
    labels_arr = np.asarray(labels)
    if limit is None or int(limit) >= len(labels_arr):
        return np.arange(len(labels_arr))
    limit = int(limit)
    classes = np.unique(labels_arr)
    per = limit // len(classes)
    extra = limit - per * len(classes)
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for i, cls in enumerate(classes):
        members = np.flatnonzero(labels_arr == cls)
        take = min(len(members), per + (1 if i < extra else 0))
        parts.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def _find_parquet_shards(root: Path, split: str) -> list[Path]:
    for candidate in (root / "data", root):
        if candidate.is_dir():
            shards = sorted(candidate.glob(f"{split}-*.parquet"))
            if shards:
                return shards
    return []


class ParquetImageDataset(Dataset):
    """Reads an HF-style parquet dump directly from a directory.

    Shards live at <root>/data/<split>-00000-of-000NN.parquet (or
    <root>/<split>-*.parquet). Label columns are loaded eagerly (tiny); image
    bytes are decoded lazily with an LRU cache of whole shards so repeat
    access to the same shard is cheap.
    """

    def __init__(self, root: str | os.PathLike, split: str, transform: Callable | None = None,
                 limit: int | None = None, holdout_generator: str | None = None):
        import pyarrow.parquet as pq  # lazy import: only needed for this source

        root = Path(root)
        split = canonical_split(split)
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.transform = transform
        self.samples: list[tuple[int, int, int]] = []  # (shard_idx, row, label_b)
        holdout_id = _holdout_id(holdout_generator)
        self.shards = _find_parquet_shards(root, split)
        if not self.shards:
            raise FileNotFoundError(
                f"No parquet shards for split '{split}' under {root} "
                f"(expected <root>/data/{split}-*.parquet or <root>/{split}-*.parquet)")
        self._pq = pq
        self._cache: dict[int, tuple[list, list, list]] = {}
        self._cache_order: list[int] = []
        self._max_cached_shards = 2
        for shard_idx, shard in enumerate(self.shards):
            table = pq.read_table(shard, columns=["Label_A", "Label_B"])
            la_values = table.column("Label_A").to_pylist()
            lb_values = table.column("Label_B").to_pylist()
            for row, (la, lb) in enumerate(zip(la_values, lb_values)):
                if holdout_id is not None and int(lb) == holdout_id:
                    continue
                self.samples.append((shard_idx, row, int(lb)))
        if limit:
            labels = [s[2] for s in self.samples]
            idx = stratified_limit_indices(labels, limit, seed=SEED)
            self.samples = [self.samples[i] for i in idx]

    # ------------------------------------------------------------- internals
    def _load_shard(self, shard_idx: int):
        cached = self._cache.get(shard_idx)
        if cached is not None:
            return cached
        table = self._pq.read_table(self.shards[shard_idx],
                                    columns=["Image", "Label_A", "Label_B"])
        images = table.column("Image").to_pylist()
        la = table.column("Label_A").to_pylist()
        lb = table.column("Label_B").to_pylist()
        loaded = (images, la, lb)
        self._cache[shard_idx] = loaded
        self._cache_order.append(shard_idx)
        if len(self._cache_order) > self._max_cached_shards:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return loaded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        shard_idx, row, label_b = self.samples[index]
        images = self._load_shard(shard_idx)[0]
        img = _decode_parquet_image(images[row])
        label_a = 0 if label_b == 0 else 1
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label_a": label_a, "label_b": label_b, "index": index}

    def labels_b(self) -> np.ndarray:
        return np.array([s[2] for s in self.samples], dtype=np.int64)

    def raw_image(self, index: int):
        shard_idx, row, label_b = self.samples[int(index)]
        images = self._load_shard(shard_idx)[0]
        return _decode_parquet_image(images[row])


def _decode_parquet_image(raw):
    if isinstance(raw, dict):
        data = raw.get("bytes")
        if data is None:
            path = raw.get("path")
            if isinstance(path, str) and Path(path).is_file():
                data = Path(path).read_bytes()
        if data is None:
            raise ValueError("parquet Image cell contains neither bytes nor a readable path")
    elif isinstance(raw, (bytes, bytearray)):
        data = raw
    elif isinstance(raw, Image.Image):
        return raw.convert("RGB")
    else:
        raise ValueError(f"unsupported Image cell type: {type(raw)}")
    return Image.open(io.BytesIO(data)).convert("RGB")


def collate_dict(batch: list[dict]) -> dict:
    out: dict = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["label_a"] = torch.tensor([int(b["label_a"]) for b in batch], dtype=torch.long)
    out["label_b"] = torch.tensor([int(b["label_b"]) for b in batch], dtype=torch.long)
    out["index"] = torch.tensor([int(b["index"]) for b in batch], dtype=torch.long)
    return out


def compute_pos_weight(labels_a: np.ndarray) -> float:
    """pos_weight = #negatives / #positives for BCEWithLogits (fake = positive class)."""
    labels_a = np.asarray(labels_a, dtype=np.int64)
    n_pos = int((labels_a == 1).sum())
    n_neg = int((labels_a == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def compute_attr_class_weights(labels_b: np.ndarray) -> np.ndarray:
    """w_k = N / (K * n_k), per the MS-COCOAI protocol (K = 6)."""
    labels_b = np.asarray(labels_b, dtype=np.int64)
    n = len(labels_b)
    k = len(LABEL_B_NAMES)
    counts = np.bincount(labels_b, minlength=k).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return n / (k * counts)
