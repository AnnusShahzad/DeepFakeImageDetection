import random

import numpy as np
import pytest
import torch
from PIL import Image

from mscocoai.constants import LABEL_B_NAMES
from mscocoai.data import (
    LocalFolderDataset, RandomJPEGCompression, build_dataset, build_loader,
    collate_dict, compute_attr_class_weights, compute_pos_weight, get_transforms,
)


def test_label_mapping_and_labels_a(tiny_dataset):
    ds = LocalFolderDataset(tiny_dataset, "train", transform=None)
    by_name = {}
    for path, label_b in ds.samples:
        by_name.setdefault(path.parent.name, label_b)
    assert by_name == {name: i for i, name in enumerate(LABEL_B_NAMES)}
    item = next(iter(ds))
    assert item["label_a"] == (0 if item["label_b"] == 0 else 1)


def test_split_sizes(tiny_dataset):
    for split in ("train", "validation", "test"):
        ds = LocalFolderDataset(tiny_dataset, split)
        assert len(ds) == 4 * len(LABEL_B_NAMES)


def test_missing_split_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalFolderDataset(tmp_path, "train")


def test_jpeg_aug_outputs_rgb(tiny_dataset):
    img = Image.open(next((tiny_dataset / "train" / "sd3").glob("*.jpg"))).convert("RGB")
    out = RandomJPEGCompression(quality=(70, 70), p=1.0)(img)
    assert isinstance(out, Image.Image) and out.mode == "RGB" and out.size == img.size


def test_transform_shapes(tiny_dataset):
    ds = LocalFolderDataset(tiny_dataset, "train", transform=get_transforms("train", img_size=64))
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    ds_test = LocalFolderDataset(tiny_dataset, "test", transform=get_transforms("test", img_size=64))
    assert ds_test[0]["image"].shape == (3, 64, 64)


def test_degradation_jpeg75(tiny_dataset):
    ds = LocalFolderDataset(tiny_dataset, "test", transform=get_transforms("test", img_size=64, degradation="jpeg75"))
    assert ds[0]["image"].shape == (3, 64, 64)


def test_pos_weight_five_to_one():
    labels = np.array([0] * 16 + [1] * 80)
    assert compute_pos_weight(labels) == pytest.approx(0.2)


def test_attr_class_weights_balanced():
    labels = np.repeat(np.arange(6), 100)
    w = compute_attr_class_weights(labels)
    assert np.allclose(w, np.ones(6))


def test_collate_dict(tiny_dataset):
    ds = LocalFolderDataset(tiny_dataset, "train", transform=get_transforms("train", img_size=64))
    batch = collate_dict([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["label_a"].dtype == torch.long and batch["label_b"].dtype == torch.long


def test_build_loader_and_deterministic_limit(tiny_dataset):
    ds = build_dataset("train", data_dir=str(tiny_dataset), limit=10, norm="imagenet", img_size=64)
    loader = build_loader(ds, batch_size=4, shuffle=False, num_workers=0)
    n = sum(batch["image"].shape[0] for batch in loader)
    assert n == 10


import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

import mscocoai.data as data_mod


def _jpg_bytes(color) -> bytes:
    img = PILImage.new("RGB", (16, 16), color)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_parquet_shard(path) -> None:
    dtype = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])
    rows = [
        {"bytes": _jpg_bytes((10, 10, 10)), "path": "a.jpg"},   # Label_B=0 real
        {"bytes": _jpg_bytes((20, 20, 20)), "path": "b.jpg"},   # Label_B=1 sd21
        {"bytes": _jpg_bytes((30, 30, 30)), "path": "c.jpg"},   # Label_B=2 sdxl
        {"bytes": _jpg_bytes((40, 40, 40)), "path": "d.jpg"},   # Label_B=3 sd3
        {"bytes": _jpg_bytes((50, 50, 50)), "path": "e.jpg"},   # Label_B=4 dalle3
        {"bytes": _jpg_bytes((60, 60, 60)), "path": "f.jpg"},   # Label_B=5 mj6
    ] * 2
    img_arr = pa.array(rows, type=dtype)
    table = pa.table({
        "Image": img_arr,
        "Label_A": pa.array([0, 1, 1, 1, 1, 1] * 2, type=pa.int32()),
        "Label_B": pa.array([0, 1, 2, 3, 4, 5] * 2, type=pa.int32()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_detect_layout_identifies_parquet(tmp_path):
    assert data_mod.detect_layout(tmp_path) == "unknown"
    root = tmp_path / "pq"
    _make_parquet_shard(root / "data" / "train-00000-of-00001.parquet")
    assert data_mod.detect_layout(root) == "parquet"
    bare = tmp_path / "pq2"
    _make_parquet_shard(bare / "train-00000-of-00001.parquet")
    assert data_mod.detect_layout(bare) == "parquet"


def test_detect_layout_identifies_folder(tmp_path):
    root = tmp_path / "folder"
    (root / "train").mkdir(parents=True)
    assert data_mod.detect_layout(root) == "folder"


def test_parquet_dataset_reads_images_and_labels(tmp_path):
    root = tmp_path / "pqds"
    _make_parquet_shard(root / "data" / "train-00000-of-00001.parquet")
    ds = data_mod.ParquetImageDataset(root, "train", transform=None)
    assert len(ds) == 12
    item = ds[0]
    assert {"image", "label_a", "label_b", "index"} <= set(item)
    labels = sorted(set(ds.labels_b().tolist()))
    assert labels == [0, 1, 2, 3, 4, 5]


def test_parquet_dataset_holdout_generator(tmp_path):
    root = tmp_path / "pqh"
    _make_parquet_shard(root / "data" / "train-00000-of-00001.parquet")
    ds = data_mod.ParquetImageDataset(root, "train", transform=None, holdout_generator="sd3")
    assert 3 not in set(ds.labels_b().tolist())
    assert 0 in set(ds.labels_b().tolist())


def test_folder_dataset_holdout_generator(tiny_dataset):
    ds = data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=None,
                                norm="imagenet", img_size=64, holdout_generator="sd3")
    assert 3 not in set(ds.labels_b().tolist())
    assert len(ds.labels_b()) == 20   # 24 originals minus 4 sd3


def test_holdout_generator_real_is_rejected(tiny_dataset):
    with pytest.raises(ValueError):
        data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=None,
                               norm="imagenet", img_size=64, holdout_generator="real")


def test_holdout_generator_unknown_name_rejected(tiny_dataset):
    with pytest.raises(ValueError):
        data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=None,
                               norm="imagenet", img_size=64, holdout_generator="banana")


def test_stratified_limit_is_class_balanced(tiny_dataset):
    ds = data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=12,
                                norm="imagenet", img_size=64)
    labels = ds.labels_b()
    counts = {name: int((labels == i).sum()) for i, name in enumerate(LABEL_B_NAMES)}
    assert counts == {name: 2 for name in LABEL_B_NAMES}


def test_stratified_limit_covers_all_classes_with_odd_limit(tiny_dataset):
    ds = data_mod.build_dataset("train", data_dir=str(tiny_dataset), limit=7,
                                norm="imagenet", img_size=64)
    labels = set(ds.labels_b().tolist())
    assert labels == set(range(6))   # every class survives a tiny smoke limit


def test_stratified_limit_indices_deterministic():
    labels = [0] * 50 + [1] * 50 + [2] * 50
    a = data_mod.stratified_limit_indices(labels, 30, seed=0)
    b = data_mod.stratified_limit_indices(labels, 30, seed=0)
    assert list(a) == list(b)
    assert sum(np.asarray(labels)[a] == 0) == 10
