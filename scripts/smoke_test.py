#!/usr/bin/env python
"""Offline end-to-end check on a tiny synthetic Food-101 (no download, no pretrained weights).

1. unit checks: pHash distances, top-k / ECE, every config builds (forward + backward),
   frozen-backbone mode trains only the head and keeps BatchNorm in eval mode
2. pipeline: fake archive (101 classes, planted near-duplicates) -> preresize -> index_images ->
   make_splits -> eda -> noise_audit make/score -> train (B0, P1, E1; a few steps) -> resume ->
   evaluate -> compare
Usage:  python scripts/smoke_test.py [--keep] [--skip-pipeline]
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from src.data.phash import near_duplicate_pairs, phash
from src.engine.metrics import expected_calibration_error, top_k_accuracy
from src.models import build_model, count_macs, count_parameters
from src.utils.config import load_config


def check_units():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).resize((256, 256))
    h1, h2 = phash(img), phash(img.resize((200, 200)))
    h3 = phash(Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)))
    a = np.array([h1], dtype=np.uint64)
    i, j, d = near_duplicate_pairs(a, np.array([h2, h3], dtype=np.uint64), 4)
    assert list(j) == [0], (j, d)
    print(f"[ok] pHash: resized copy found (d={d[0]}), random image not")

    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 1])
    assert abs(top_k_accuracy(y, probs, 1) - 1 / 3) < 1e-9 and top_k_accuracy(y, probs, 2) == 1.0
    assert 0 <= expected_calibration_error(y, probs)["ece"] <= 1
    print("[ok] top-k accuracy and ECE")


def check_models():
    cfg_files = sorted(p for p in (ROOT / "configs" / "food101").glob("*.yaml") if not p.name.startswith("_"))
    cfg_files += sorted((ROOT / "configs" / "food101" / "ablations").glob("*.yaml"))
    criterion = nn.CrossEntropyLoss()
    names = set()
    print(f"\n{'experiment':34s} {'group':9s} {'params':>12s} {'trainable':>12s} {'GMACs':>7s}")
    for path in cfg_files:
        cfg = load_config(path)
        mcfg = dict(cfg["model"])
        if mcfg["name"] != "simple_cnn":
            mcfg["pretrained"] = False
        model = build_model(mcfg, 101)
        x = torch.randn(2, 3, 224, 224)
        model.train()
        logits = model(x)
        assert logits.shape == (2, 101), logits.shape
        criterion(logits, torch.randint(0, 101, (2,))).backward()
        n = count_parameters(model)
        exp = cfg["experiment"]
        names.add(exp["name"])
        if mcfg.get("freeze_backbone"):
            assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith(model.head_prefix))
            assert n["trainable"] == 2048 * 101 + 101, n
            bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
            assert bns and not any(m.training for m in bns), "frozen backbone BN must stay in eval mode"
        print(f"{exp['name']:34s} {exp.get('group', 'main'):9s} {n['total']:>12,d} {n['trainable']:>12,d} "
              f"{count_macs(model) / 1e9:7.2f}")
    for path in cfg_files:
        ref = load_config(path)["experiment"].get("reference")
        assert ref is None or ref in names, f"reference '{ref}' not found for {path}"
    print("[ok] all configs build, forward and backward; linear probe trains only the head")


def make_fake_food101(root: Path, n_train=10, n_test=3, seed=0):
    from torchvision.datasets import Food101
    classes = sorted(Food101.__dict__.get("_CLASSES", [])) or [f"class_{i:03d}" for i in range(101)]
    rng = np.random.default_rng(seed)
    meta = root / "meta"
    meta.mkdir(parents=True)
    train, test = [], []
    sizes = [(512, 384), (384, 512), (512, 512), (300, 200)]
    for ci, c in enumerate(classes):
        (root / "images" / c).mkdir(parents=True)
        base = rng.integers(0, 255, 3)
        for k in range(n_train + n_test):
            iid = f"{ci * 1000 + k}"
            w, h = sizes[rng.integers(len(sizes))]
            im = Image.new("RGB", (w, h), tuple(int(v) for v in base))
            d = ImageDraw.Draw(im)
            for _ in range(6):
                x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
                d.ellipse([x0, y0, x0 + int(rng.integers(20, w // 2)), y0 + int(rng.integers(20, h // 2))],
                          fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
            if ci == 0 and k == n_train:            # plant a test near-duplicate of a train image
                im = Image.open(root / "images" / c / f"{train[0].split('/')[1]}.jpg").resize((w, h))
            if ci == 1 and k == 0:                  # plant a grayscale image
                im = im.convert("L")
            im.save(root / "images" / c / f"{iid}.jpg", quality=90)
            (train if k < n_train else test).append(f"{c}/{iid}")
    (meta / "classes.txt").write_text("\n".join(classes) + "\n")
    (meta / "train.txt").write_text("\n".join(train) + "\n")
    (meta / "test.txt").write_text("\n".join(test) + "\n")
    return classes


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd))
    rc = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT).returncode
    assert rc == 0, f"command failed with exit code {rc}"


def check_pipeline(work: Path):
    raw, resized = work / "raw" / "food-101", work / "food-101-256"
    make_fake_food101(raw)
    splits, index, outputs = work / "splits", work / "meta" / "image_index.csv.gz", work / "outputs"
    data = [f"data.image_root={resized}", f"data.raw_root={raw}", f"data.split_dir={splits}",
            f"data.index_file={index}", "data.num_workers=0"]
    run(["scripts/preresize.py", "--src", raw, "--dst", resized, "--workers", "2"])
    run(["scripts/index_images.py", "--image-root", resized, "--raw-root", raw, "--out", index, "--workers", "2"])
    run(["scripts/make_splits.py", "--image-root", resized, "--index", index, "--split-dir", splits])
    run(["scripts/eda.py", "--set", *data, "--out", outputs / "eda"])
    audit_csv = work / "label_audit.csv"
    cfg_path = work / "smoke_base.yaml"
    cfg_path.write_text(f"base: {ROOT / 'configs' / 'food101' / '_base.yaml'}\ndata:\n  image_root: {resized}\n"
                        f"  split_dir: {splits}\n")
    run(["scripts/noise_audit.py", "make", "--config", cfg_path, "--csv", audit_csv, "--out", outputs / "audit",
         "--n-classes", "3", "--per-class", "4"])
    df = pd.read_csv(audit_csv)
    df["verdict"] = ["ok"] * (len(df) - 2) + ["wrong", "unsure"]
    df.to_csv(audit_csv, index=False)
    run(["scripts/noise_audit.py", "score", "--config", cfg_path, "--csv", audit_csv, "--out", outputs / "audit"])

    fast = [*data, f"output.root={outputs / 'runs'}", "train.batch_size=8", "train.eval_batch_size=16",
            "train.epochs=2", "train.max_steps_per_epoch=3", "train.show_progress=false",
            "eval.bootstrap_samples=50", "eval.n_examples=5", "eval.inference.batch_size=8",
            "eval.inference.timed_iters=2", "eval.inference.latency_samples=3", "eval.inference.warmup_iters=1"]
    run(["scripts/train.py", "--config", "configs/food101/b0_simple_cnn.yaml", "--set", *fast,
         "data.train_fraction=0.5"])
    # interrupted run: 1 epoch, then resume to epoch 2
    run(["scripts/train.py", "--config", "configs/food101/p1_resnet50.yaml", "--no-eval", "--set", *fast,
         "model.pretrained=false", "train.epochs=2", "train.max_hours=0.000001"])
    run(["scripts/train.py", "--config", "configs/food101/p1_resnet50.yaml", "--resume", "--set", *fast,
         "model.pretrained=false", "train.max_hours=0"])
    hist = pd.read_csv(outputs / "runs" / "food101" / "p1_resnet50" / "seed42" / "history.csv")
    assert list(hist.epoch) == [1, 2], hist.epoch.tolist()
    run(["scripts/train.py", "--config", "configs/food101/ablations/e1_resnet50_linear_probe.yaml",
         "--set", *fast, "model.pretrained=false"])
    run(["scripts/evaluate.py", "--run-dir", outputs / "runs" / "food101" / "b0_simple_cnn" / "seed42",
         "--split", "val"])
    run(["scripts/compare.py", "--runs-root", outputs / "runs", "--out", outputs / "comparison"])
    for f in ["comparison/food101/summary.md", "comparison/food101/decisions.md", "eda/eda_summary.json",
              "eda/samples_all_classes.png", "audit/audit_result.json",
              "runs/food101/p1_resnet50/seed42/eval/test_metrics.json"]:
        assert (outputs / f).exists(), f
    print("\n[ok] end-to-end pipeline on synthetic data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", action="store_true", help="keep the temporary work folder")
    p.add_argument("--skip-pipeline", action="store_true")
    args = p.parse_args()
    torch.manual_seed(0)
    check_units()
    check_models()
    if args.skip_pipeline:
        return
    work = Path(tempfile.mkdtemp(prefix="food101_smoke_"))
    try:
        check_pipeline(work)
    finally:
        if args.keep:
            print(f"work folder kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
