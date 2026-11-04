"""Downloader: export-to-mirror path (network parts are exercised on the server)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mscocoai import download


def _jpg_bytes(color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", (16, 16), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_shard(path: Path, labels: list[int], split: str = "train") -> Path:
    dtype = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])
    img_arr = pa.array([{"bytes": _jpg_bytes((i * 10, i * 5, i * 7)), "path": f"{i}.jpg"}
                        for i in range(len(labels))], type=dtype)
    table = pa.table({
        "Image": img_arr,
        "Label_A": pa.array([0 if b == 0 else 1 for b in labels], type=pa.int32()),
        "Label_B": pa.array(labels, type=pa.int32()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def test_sniff_ext_detects_common_formats():
    assert download._sniff_ext(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert download._sniff_ext(b"\x89PNG\r\n\x1a\nrest") == ".png"
    assert download._sniff_ext(b"RIFF\x00\x00\x00\x00WEBP") == ".webp"
    assert download._sniff_ext(b"garbage") == ".bin"


def test_export_images_writes_class_folders(tmp_path):
    shard = _make_shard(tmp_path / "data" / "train-00000-of-00001.parquet", [1, 2, 2])
    dest = tmp_path / "mirror"
    summary = download.export_images([shard], dest, workers=1)
    assert summary["written"] == 3
    assert (dest / "train" / "sd21").is_dir()
    assert (dest / "train" / "sdxl").is_dir()
    assert len(list((dest / "train" / "sdxl").glob("*.jpg"))) == 2
    assert (dest / "manifest.json").is_file()


def test_export_images_is_resumable(tmp_path):
    shard = _make_shard(tmp_path / "data" / "validation-00000-of-00001.parquet", [0, 3, 4])
    dest = tmp_path / "mirror"
    first = download.export_images([shard], dest, workers=1)
    assert first["written"] == 3
    again = download.export_images([shard], dest, workers=1)
    assert again["written"] == 0 and again["skipped"] == 3


def test_split_from_shard_name():
    assert download._split_from_shard("train-00000-of-00007.parquet") == "train"
    assert download._split_from_shard("validation-00001-of-00002.parquet") == "validation"
    assert download._split_from_shard("test-00003-of-00008.parquet") == "test"
