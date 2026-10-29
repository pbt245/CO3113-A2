#!/usr/bin/env python
"""Build the per-image index (one decode per image) used by make_splits.py and eda.py.

For every official train/test image: raw width/height/mode/file size (header only), pre-resized
width/height/file size, per-channel pixel sums (-> train-split normalization), brightness,
contrast, saturation and a 64-bit pHash (-> near-duplicate check).

Usage:  python scripts/index_images.py --workers 8
Output: data/meta/image_index.csv.gz (committed; ~101k rows)
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.data.food101 import entry_label, index_one, read_meta
from src.utils.io import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="data/resized/food-101-256")
    p.add_argument("--raw-root", default="data/raw/food-101", help="'none' to skip raw-size columns")
    p.add_argument("--out", default="data/meta/image_index.csv.gz")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    image_root = resolve_path(args.image_root)
    raw_root = None if args.raw_root.lower() == "none" else resolve_path(args.raw_root)
    classes, train, test = read_meta(image_root)
    jobs = [(e, image_root, raw_root) for e in train + test]
    with Pool(args.workers) as pool:
        rows = list(tqdm(pool.imap(index_one, jobs, chunksize=64), total=len(jobs), desc="index"))
    df = pd.DataFrame(rows)
    df.insert(1, "label", entry_label(df["entry"].tolist(), classes))
    df.insert(1, "class", [classes[i] for i in df["label"]])
    df.insert(1, "official_split", ["train"] * len(train) + ["test"] * len(test))
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"{len(df)} images indexed -> {out} ({out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
