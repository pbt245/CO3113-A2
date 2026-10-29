#!/usr/bin/env python
"""(Re-)evaluate the best checkpoint of a finished run.

Usage:  python scripts/evaluate.py --run-dir outputs/runs/food101/p1_resnet50/seed42 [--split test|val]
        [--set data.image_root=/path/to/food-101-256]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.evaluate import evaluate_run
from src.utils.env import resolve_device
from src.utils.io import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], help="e.g. data.image_root=... on another machine")
    args = p.parse_args()
    evaluate_run(resolve_path(args.run_dir), resolve_device(args.device), split=args.split, overrides=args.set)


if __name__ == "__main__":
    main()
