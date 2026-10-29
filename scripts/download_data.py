#!/usr/bin/env python
"""Download / extract the official Food-101 archive into data/raw/food-101.

Usage:  python scripts/download_data.py                 # torchvision download (~5 GB, md5-checked)
        python scripts/download_data.py --archive data/raw/food-101.tar.gz   # extract a local copy
Result: data/raw/food-101/{images/<class>/<id>.jpg, meta/{classes,train,test}.txt}
"""
import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.food101 import read_meta
from src.utils.io import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--archive", default=None, help="extract this food-101.tar.gz instead of downloading")
    args = p.parse_args()
    raw = resolve_path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "food-101" / "meta" / "classes.txt").exists():
        print(f"Already present: {raw / 'food-101'}")
    elif args.archive:
        with tarfile.open(resolve_path(args.archive)) as tar:
            try:
                tar.extractall(raw, filter="data")
            except TypeError:                     # Python < 3.11.4 has no extraction filters
                tar.extractall(raw)
    else:
        from torchvision.datasets import Food101
        Food101(root=str(raw), split="train", download=True)
    classes, train, test = read_meta(raw / "food-101")
    print(f"Food-101 at {raw / 'food-101'}: {len(classes)} classes, {len(train)} train / {len(test)} test entries")


if __name__ == "__main__":
    main()
