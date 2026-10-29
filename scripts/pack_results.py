#!/usr/bin/env python
"""Zip EDA, comparison outputs and run folders to move results between machines (Kaggle -> repo).

Usage:  python scripts/pack_results.py [--with-checkpoints best|all] [--out-dir outputs]
Output: <out-dir>/results_bundle_<timestamp>.zip  (paths relative to the repo root, unzip at the root)
Checkpoints are excluded by default (best.pt of ResNet-50 is ~95 MB; last.pt also holds optimizer state).
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-checkpoints", choices=["none", "best", "all"], default="none")
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"results_bundle_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    n_files = 0
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in [OUT / "eda", OUT / "comparison", OUT / "runs"]:
            if not root.exists():
                print(f"(missing) {root.relative_to(ROOT)}")
                continue
            for f in root.rglob("*"):
                if f.is_dir() or f.suffix == ".zip":
                    continue
                if "checkpoints" in f.parts and not (args.with_checkpoints == "all" or
                                                     (args.with_checkpoints == "best" and f.name == "best.pt")):
                    continue
                zf.write(f, f.relative_to(ROOT))
                n_files += 1
    print(f"{n_files} files -> {bundle} ({bundle.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
