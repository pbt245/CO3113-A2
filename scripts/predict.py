#!/usr/bin/env python
"""Top-5 prediction for any image file with a finished run's best checkpoint (demo).

  python scripts/predict.py photo.jpg [more.jpg ...] --run-dir outputs/runs/food101/p3_convnext_tiny/seed42
Uses the run's eval transform (Resize 256 + CenterCrop 224 + its normalization).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.data.food101 import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_split
from src.models import build_model
from src.utils.config import load_config
from src.utils.env import resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--run-dir", default="outputs/runs/food101/p1_resnet50/seed42")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    run_dir = ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    split, _ = load_split(cfg["data"]["split_dir"])
    classes = split["classes"]
    if cfg["data"].get("normalization", "imagenet") == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = split["normalization"]["mean"], split["normalization"]["std"]
    tf = build_transforms(cfg["data"], mean, std, train=False)

    model_cfg = dict(cfg["model"])
    if model_cfg["name"] != "simple_cnn":
        model_cfg["pretrained"] = False
    model = build_model(model_cfg, len(classes))
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(args.device)
    model.to(device).eval()

    print(f"model {cfg['experiment']['name']} seed {cfg['seed']} (checkpoint epoch {ckpt.get('epoch')})")
    for f in args.images:
        with Image.open(f) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(x).softmax(1)[0].cpu()
        top = probs.topk(5)
        print(f"\n{f}")
        for prob, idx in zip(top.values, top.indices):
            print(f"  {classes[idx]:28s} {prob:.3f}")


if __name__ == "__main__":
    main()
