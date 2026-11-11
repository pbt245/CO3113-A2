#!/usr/bin/env python
"""Zip EDA, comparison outputs and run folders to move results between machines (Kaggle -> repo).

Usage:  python scripts/pack_results.py [--with-checkpoints best|all] [--out-dir outputs]
        python scripts/pack_results.py --checkpoints-only      # checkpoint release (best.pt of every run)
Output: <out-dir>/results_bundle_<timestamp>.zip  (paths relative to the repo root, unzip at the root)
        <out-dir>/checkpoints_food101.zip + CHECKPOINTS_SHA256.txt (same hierarchy, timing/debug runs excluded)
Checkpoints are excluded by default (best.pt of ResNet-50 is ~95 MB; last.pt also holds optimizer state).
"""
import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def pack_checkpoints(out_dir: Path):
    """best.pt of every finished run, stored as outputs/runs/food101/<experiment>/seed<seed>/checkpoints/best.pt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in (OUT / "runs").glob("*/*/seed*/checkpoints/best.pt")
                   if not f.parts[-4].startswith("timing_") and (f.parents[1] / "eval" / "test_metrics.json").exists())
    bundle = out_dir / "checkpoints_food101.zip"
    lines = []
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as zf:      # weights do not compress
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{h}  {rel}")
            zf.write(f, rel)
        manifest = "\n".join(lines) + "\n"
        zf.writestr("CHECKPOINTS_SHA256.txt", manifest)
    (out_dir / "CHECKPOINTS_SHA256.txt").write_text(manifest)
    print(f"{len(files)} checkpoints -> {bundle} ({bundle.stat().st_size / 1024**2:.0f} MB)\n"
          f"Unzip at the repo root; check with: sha256sum -c CHECKPOINTS_SHA256.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-checkpoints", choices=["none", "best", "all"], default="none")
    p.add_argument("--checkpoints-only", action="store_true")
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()
    if args.checkpoints_only:
        return pack_checkpoints(Path(args.out_dir))

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
