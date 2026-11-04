import io
import random
from pathlib import Path

import pytest
from PIL import Image

from mscocoai.constants import LABEL_B_NAMES

IMG_EXTS = (".jpg",)


def _write_image(path: Path, size: int, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.frombytes("RGB", (size, size), bytes(rng.randrange(256) for _ in range(size * size * 3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=95)


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """A tiny local mirror: <root>/<split>/<label_name>/*.jpg (4 images per cell)."""
    root = tmp_path / "ds"
    for split in ("train", "validation", "test"):
        for class_id, name in enumerate(LABEL_B_NAMES):
            for i in range(4):
                seed = hash((split, class_id, i)) & 0xFFFF
                _write_image(root / split / name / f"{i}.jpg", size=64, seed=seed)
    return root
