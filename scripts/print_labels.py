#!/usr/bin/env python3
"""Print the exact MS-COCOAI label mapping and split sizes.

No dataset download needed for the mapping table. With --check-hf it opens a
streaming connection to the HF dataset to confirm one sample per split
(does NOT download the 7.5 GB data).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mscocoai.constants import (
    HF_DATASET_ID, LABEL_A_NAMES, LABEL_B_NAMES, SPLIT_SIZES,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hf", action="store_true",
                        help="stream one sample per split from Hugging Face (small download)")
    args = parser.parse_args()

    print(f"Dataset: {HF_DATASET_ID}")
    print(f"Splits (images): {SPLIT_SIZES}")
    print()
    print("Label_A (binary):")
    for i, name in enumerate(LABEL_A_NAMES):
        print(f"  {i} = {name}")
    print()
    print("Label_B (attribution, 6 classes):")
    for i, name in enumerate(LABEL_B_NAMES):
        print(f"  {i} = {name}")
    print()
    print("All classes are 16,000 images per split-source; Label_A is 1:5 real:fake.")
    print("Official splits: train 42,000 / validation 9,000 / test 45,000.")

    if args.check_hf:
        from datasets import load_dataset
        for split in ("train", "validation", "test"):
            ds = load_dataset(HF_DATASET_ID, split=split, streaming=True)
            row = next(iter(ds))
            print(f"\n[{split}] Label_A={row['Label_A']} Label_B={row['Label_B']} "
                  f"caption={row['Caption']!r}")


if __name__ == "__main__":
    main()
