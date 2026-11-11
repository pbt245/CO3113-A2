#!/usr/bin/env python
"""Grad-CAM panels for selected test images [OPTIONAL interpretability extension].

  python scripts/gradcam.py                     # P1 and P2 (seed 42), default selection below
  python scripts/gradcam.py --models b0_simple_cnn p1_resnet50 p2_efficientnet_b0 p3_convnext_tiny

Selection (from outputs/error_analysis/food101/errors_p1_resnet50_seed42.csv, written by
`scripts/error_analysis.py taxonomy`; seed 42): the most confident P1 errors of each bucket
(sibling / consensus / near-miss / gross) and the same number of confident correct predictions.
Each row: the eval-time centre crop, then each model's Grad-CAM for the class that model predicts.
Output: outputs/error_analysis/food101/gradcam.png (+ gradcam_selection.csv)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.engine.evaluate import load_run
from src.utils.env import resolve_device
from src.utils.io import resolve_path
from src.viz.gradcam import GradCAM, overlay

SHORT = {"b0_simple_cnn": "B0", "p1_resnet50": "P1", "p2_efficientnet_b0": "P2", "p3_convnext_tiny": "P3"}


def select(errors_csv: Path, per_bucket: int, n_correct: int) -> pd.DataFrame:
    df = pd.read_csv(errors_csv, keep_default_na=False)
    picks = []
    for b in ["consensus", "sibling", "near_miss", "gross"]:
        g = df[(df.bucket == b)].sort_values("confidence", ascending=False).head(per_bucket)
        picks.append(g.assign(reason=f"P1 error: {b}"))
    ok = df[df.correct.astype(str) == "True"]
    picks.append(ok[ok.confidence > 0.9].sample(n_correct, random_state=42).assign(reason="P1 correct"))
    return pd.concat(picks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["p1_resnet50", "p2_efficientnet_b0"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-bucket", type=int, default=3)
    p.add_argument("--n-correct", type=int, default=4)
    p.add_argument("--out", default="outputs/error_analysis/food101")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], help="e.g. data.image_root=... on another machine")
    args = p.parse_args()

    out = resolve_path(args.out)
    sel = select(out / f"errors_p1_resnet50_seed{args.seed}.csv", args.per_bucket, args.n_correct)
    sel.to_csv(out / "gradcam_selection.csv", index=False)
    device = resolve_device(args.device)

    cams = {}
    dataset = classes = None
    for exp in args.models:
        run_dir = resolve_path("outputs/runs") / "food101" / exp / f"seed{args.seed}"
        cfg, model, _, _, datasets, meta = load_run(run_dir, device, overrides=args.set)
        model.float()
        dataset, classes = datasets["test"], meta["classes"]
        gc = GradCAM(model)
        cams[exp] = []
        for idx in sel["index"]:
            x = dataset[int(idx)][0].unsqueeze(0).to(device)
            cams[exp].append(gc(x))
        gc.remove()
        del model

    n, k = len(sel), len(args.models) + 1
    fig, axes = plt.subplots(n, k, figsize=(2.6 * k, 2.75 * n))
    for r, (_, row) in enumerate(sel.iterrows()):
        img = dataset.display_image(int(row["index"]))
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"label: {row.true_name}\n[{row.reason}]", fontsize=7)
        for c, exp in enumerate(args.models, 1):
            cam, cls, prob = cams[exp][r]
            ok = cls == int(row.y_true)
            axes[r, c].imshow(overlay(img, cam))
            axes[r, c].set_title(f"{SHORT.get(exp, exp)}: {classes[cls]} ({prob:.2f})", fontsize=7,
                                 color="tab:green" if ok else "tab:red")
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM for the predicted class (test set, seed {args.seed}); green = correct, red = wrong",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / n))
    fig.savefig(out / "gradcam.png", dpi=130)
    plt.close(fig)
    print(f"{n} images x {len(args.models)} models -> {out / 'gradcam.png'}")


if __name__ == "__main__":
    main()
