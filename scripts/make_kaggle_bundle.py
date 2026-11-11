#!/usr/bin/env python
"""Build the two Kaggle datasets needed by notebooks/kaggle_train.ipynb with Internet OFF.

  python scripts/make_kaggle_bundle.py code      # -> outputs/kaggle_bundle/code/   (commit first!)
  python scripts/make_kaggle_bundle.py data      # -> outputs/kaggle_bundle/data/   (once)
  then:  kaggle datasets create  -p outputs/kaggle_bundle/code     (later updates: kaggle datasets version -p ... -m "msg")
         kaggle datasets create  -p outputs/kaggle_bundle/data

code: co3113-a2-code.zip = `git archive HEAD` without outputs/ (otherwise the committed M2 runs would be skipped as
      "finished") + GIT_COMMIT (recorded in every run's env.json), and the torchvision ImageNet weights of the three
      backbones from the local torch hub cache (offline notebooks cannot download them).
data: food-101-256.zip = data/resized/food-101-256 (stored, JPEGs do not compress).
All files are flat (no sub-folders), so the Kaggle CLI uploads them as they are; private datasets by default.
"""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.pretrained import BACKBONES, _weights

EXCLUDE = ("outputs/", "notebooks/", "docs/")


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def metadata(out: Path, user: str, slug: str, title: str):
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1))


def build_code(out: Path, user: str, allow_dirty: bool):
    if git("status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        sys.exit("Tracked files have uncommitted changes: commit (and push) first, so GIT_COMMIT is exact.")
    commit = git("rev-parse", "HEAD")
    out.mkdir(parents=True, exist_ok=True)
    tar = tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", "HEAD"], cwd=ROOT)))
    n = 0
    with zipfile.ZipFile(out / "co3113-a2-code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for m in tar.getmembers():
            if m.isfile() and not m.name.startswith(EXCLUDE):
                zf.writestr(f"CO3113-A2/{m.name}", tar.extractfile(m).read())
                n += 1
        zf.writestr("CO3113-A2/GIT_COMMIT", commit + "\n")
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    for arch, (_, spec, _) in BACKBONES.items():
        name = Path(_weights(spec).url).name
        src = hub / name
        if not src.exists():                      # download once on this (online) machine
            _weights(spec).get_state_dict(progress=True)
        (out / name).write_bytes(src.read_bytes())
        print(f"weights {arch}: {name}")
    metadata(out, user, "co3113-a2-code", "co3113-a2-code")
    print(f"{n} code files at commit {commit[:10]} -> {out}")


def build_data(out: Path, user: str):
    src = ROOT / "data/resized/food-101-256"
    if not (src / "meta/classes.txt").exists():
        sys.exit(f"{src} missing: run scripts/preresize.py first")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out / "food-101-256.zip", "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent).as_posix())
                n += 1
    metadata(out, user, "food-101-256", "food-101-256")
    print(f"{n} files -> {out / 'food-101-256.zip'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["code", "data"])
    p.add_argument("--user", default="benjaminphc", help="Kaggle username (dataset owner)")
    p.add_argument("--out", default="outputs/kaggle_bundle")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    out = ROOT / args.out / args.what
    build_code(out, args.user, args.allow_dirty) if args.what == "code" else build_data(out, args.user)


if __name__ == "__main__":
    main()
