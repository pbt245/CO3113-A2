#!/usr/bin/env python
"""One-time resize of every Food-101 image to a shorter side of 256 px (JPEG quality 95).

Runs on CPU (a Kaggle CPU notebook does not use GPU quota). The training pipeline then decodes
~4x fewer pixels per image. Images whose shorter side is already <= 256 are re-encoded only
(no upscaling). Grayscale / CMYK images are converted to RGB here.

Usage:  python scripts/preresize.py --src data/raw/food-101 --dst data/resized/food-101-256 --workers 8
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from tqdm import tqdm

from src.data.food101 import image_path, read_meta
from src.utils.io import resolve_path


def resize_one(args):
    src, dst, entry, short_side, quality = args
    out = image_path(dst, entry)
    if out.exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path(src, entry)) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        if s > short_side:
            im = im.resize((round(w * short_side / s), round(h * short_side / s)), Image.BICUBIC, reducing_gap=3.0)
        im.save(out, "JPEG", quality=quality)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/food-101")
    p.add_argument("--dst", default="data/resized/food-101-256")
    p.add_argument("--short-side", type=int, default=256)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    src, dst = resolve_path(args.src), resolve_path(args.dst)
    classes, train, test = read_meta(src)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").glob("*.txt"):
        shutil.copy2(f, dst / "meta" / f.name)
    jobs = [(src, dst, e, args.short_side, args.quality) for e in train + test]
    with Pool(args.workers) as pool:
        done = sum(tqdm(pool.imap_unordered(resize_one, jobs, chunksize=64), total=len(jobs), desc="resize"))
    print(f"{done} images written, {len(jobs) - done} already present -> {dst}")


if __name__ == "__main__":
    main()
