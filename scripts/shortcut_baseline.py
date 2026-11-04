#!/usr/bin/env python3
"""Metadata-only baseline: how much of MS-COCOAI is solvable without pixels?

Every class in the published mirror has a fixed native resolution
(real 640x480, sd21 768^2, sdxl/sd3 1024^2, dalle3 270^2, mj6 436^2).
A classifier that only sees (width, height, aspect, file size) therefore
measures how much of the benchmark is a container/resolution shortcut rather
than a content-forensics problem. This is the control experiment that keeps
the headline numbers honest.

Writes <out-dir>/metadata_baseline.json and <out-dir>/metadata_baseline.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mscocoai.constants import LABEL_A_NAMES, LABEL_B_NAMES, SPLIT_SIZES

SPLITS = ("train", "validation", "test")
CLASS_DIRS = ("real", "sd21", "sdxl", "sd3", "dalle3", "mj6")
CLASS_TO_LABEL_B = {name: i for i, name in enumerate(LABEL_B_NAMES)}


def scan(split_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (features, label_a, label_b, class_names) for one split."""
    from PIL import Image

    feats, ya, yb, names = [], [], [], []
    for cls in CLASS_DIRS:
        d = split_dir / cls
        if not d.is_dir():
            continue
        lb = CLASS_TO_LABEL_B[cls]
        for fname in sorted(os.listdir(d)):
            path = d / fname
            if not path.is_file():
                continue
            size = path.stat().st_size
            try:
                with Image.open(path) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001 — unreadable files are skipped, counted below
                continue
            feats.append([w, h, w / max(h, 1), size, size / max(w * h, 1)])
            ya.append(0 if cls == "real" else 1)
            yb.append(lb)
            names.append(f"{split_dir.name}/{cls}/{fname}")
    return (np.asarray(feats, dtype=np.float64),
            np.asarray(ya, dtype=np.int64),
            np.asarray(yb, dtype=np.int64),
            names)


def fit_and_score(X_tr, y_tr, X_te, y_te, kind: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    results = {}
    scaler = StandardScaler().fit(X_tr)
    models = {
        "logreg": LogisticRegression(max_iter=2000, C=10.0),
        "tree": DecisionTreeClassifier(max_depth=8, random_state=0),
    }
    for name, model in models.items():
        Xs = scaler.transform(X_tr) if name == "logreg" else X_tr
        Xv = scaler.transform(X_te) if name == "logreg" else X_te
        model.fit(Xs, y_tr)
        pred = model.predict(Xv)
        res = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        }
        if kind == "attribution":
            res["macro_f1"] = float(f1_score(y_te, pred, average="macro", zero_division=0))
            cm = Counter(zip(y_te.tolist(), pred.tolist()))
            res["per_class_recall"] = {
                LABEL_B_NAMES[c]: round(sum(v for (t, p), v in cm.items() if t == c and p == c)
                                       / max(sum(v for (t, _), v in cm.items() if t == c), 1), 4)
                for c in range(len(LABEL_B_NAMES))
            }
        results[name] = res
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data_mirror")
    ap.add_argument("--out-dir", default="results/shortcut")
    args = ap.parse_args()

    from PIL import Image  # noqa: F401  (fail fast if Pillow is missing)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in SPLITS:
        d = data_dir / split
        if not d.is_dir():
            print(f"[skip] {d} not found")
            continue
        X, ya, yb, names = scan(d)
        splits[split] = {"X": X, "ya": ya, "yb": yb, "names": names}
        sizes = Counter((int(w), int(h)) for w, h in zip(X[:, 0], X[:, 1]))
        print(f"[scan] {split}: {len(X)} images, {len(sizes)} distinct sizes, "
              f"top: {sizes.most_common(3)}")

    if "train" not in splits or "test" not in splits:
        raise SystemExit("need both train and test splits")

    tr, te = splits["train"], splits["test"]
    report = {
        "data_dir": str(data_dir),
        "n_train": int(len(tr["X"])),
        "n_test": int(len(te["X"])),
        "features": ["width", "height", "aspect_ratio", "file_bytes", "bytes_per_pixel"],
        "detection": fit_and_score(tr["X"], tr["ya"], te["X"], te["ya"], "detection"),
        "attribution": fit_and_score(tr["X"], tr["yb"], te["X"], te["yb"], "attribution"),
        "expected_sizes": {
            "real": "640x480 (COCO native, varies)",
            "sd21": "768x768",
            "sdxl": "1024x1024",
            "sd3": "1024x1024",
            "dalle3": "270x270 (351x351 minority)",
            "mj6": "436x436",
        },
    }
    if "validation" in splits:
        va = splits["validation"]
        report["detection_validation"] = fit_and_score(tr["X"], tr["ya"], va["X"], va["ya"], "detection")
        report["attribution_validation"] = fit_and_score(tr["X"], tr["yb"], va["X"], va["yb"], "attribution")

    (out_dir / "metadata_baseline.json").write_text(json.dumps(report, indent=2))

    lines = ["# Metadata-only baseline (resolution-shortcut control)", "",
             f"Train/test: {report['n_train']} / {report['n_test']} images from `{data_dir}`.",
             "Features: " + ", ".join(report["features"]) + " — no pixels are read.", ""]
    for kind, title in (("detection", "Binary detection (Label_A)"),
                        ("attribution", "Generator attribution (Label_B)")):
        lines += [f"## {title}", "",
                  "| model | split | accuracy | balanced acc | macro F1 |",
                  "|---|---|---|---|---|"]
        for split_key, split_name in ((kind, "test"), (f"{kind}_validation", "validation")):
            if split_key not in report:
                continue
            for mname, res in report[split_key].items():
                mf1 = f"{res['macro_f1']:.4f}" if "macro_f1" in res else "—"
                lines.append(f"| {mname} | {split_name} | {res['accuracy']:.4f} | "
                             f"{res['balanced_accuracy']:.4f} | {mf1} |")
        lines.append("")
    lines += ["Interpretation: a model that never sees an image pixel but only its",
              "container metadata is a lower bound on what the benchmark allows. If this",
              "number is high, the benchmark contains a resolution shortcut and content",
              "based results must be reported alongside a resolution-normalised protocol.", ""]
    (out_dir / "metadata_baseline.md").write_text("\n".join(lines))

    det = report["detection"]["logreg"]["accuracy"]
    att = report["attribution"]["logreg"]["accuracy"]
    print(f"\n[result] metadata-only detection acc={det:.4f}  attribution acc={att:.4f}")
    print(f"[written] {out_dir}/metadata_baseline.json, {out_dir}/metadata_baseline.md")


if __name__ == "__main__":
    main()
