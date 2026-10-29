#!/usr/bin/env python
"""Create the fixed, stratified train/val split with the pHash near-duplicate checks.

Usage:  python scripts/make_splits.py [--split-seed 42 --val-fraction 0.1 --max-hamming 4] [--force]
Writes data/splits/{train,val,test,test_near_duplicates}.txt, test_near_duplicate_pairs.csv and
split.json (counts, train-only normalization, fingerprints). Commit these files.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.food101 import create_split, split_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="data/resized/food-101-256", help="only meta/ is read")
    p.add_argument("--index", default="data/meta/image_index.csv.gz")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-hamming", type=int, default=4)
    p.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = p.parse_args()

    if split_paths(args.split_dir)["json"].exists() and not args.force:
        print(f"Split already exists in {args.split_dir} (use --force to recreate)")
        return
    s, paths = create_split(args.image_root, args.split_dir, args.index, args.split_seed,
                            args.val_fraction, args.max_hamming)
    nd = s["near_duplicates"]
    print(f"train/val/test = {s['n_train']}/{s['n_val']}/{s['n_test']} | "
          f"val->train moved {nd['n_val_moved_to_train']} | test flagged {nd['n_test_flagged']} | "
          f"mean {[round(v, 4) for v in s['normalization']['mean']]} "
          f"std {[round(v, 4) for v in s['normalization']['std']]} | fingerprints {s['fingerprints']}")
    print(f"-> {paths['json'].parent}")


if __name__ == "__main__":
    main()
