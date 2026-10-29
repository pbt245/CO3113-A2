#!/usr/bin/env python
"""Task-appropriate EDA for Food-101 classification (proposal section 2.4).

Reads the image index (data/meta/image_index.csv.gz) and the committed split; loads images only
for the sample grids.
Usage:  python scripts/eda.py --config configs/food101/_base.yaml
Outputs (outputs/eda/food101/):
  eda_summary.json / eda_summary.md      every number quoted in the report
  class_counts.csv, class_distribution.png
  image_sizes.png                        raw width / height / aspect ratio / file size
  pixel_stats.png, pixel_stats.csv       per-channel mean/std, brightness / contrast / saturation per split
  class_brightness.csv                   per-class mean brightness / saturation
  samples_all_classes.png                one centre crop per class
  confusable_pairs.png                   5 images each for visually similar class pairs
  near_duplicates.csv / near_duplicates_examples.png   pHash pairs within and across splits
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T

from src.data.food101 import channel_stats, image_path, load_index, load_split
from src.data.imagenet_overlap import FOOD101_TO_IMAGENET
from src.data.phash import hex_to_uint64, near_duplicate_pairs
from src.utils.config import apply_overrides, load_config
from src.utils.io import resolve_path, save_json
from src.viz.plots import plot_image_grid

# a-priori confusable pairs (proposal section 1); M3 replaces them with the measured top confusions
CONFUSABLE_PAIRS = [
    ("steak", "filet_mignon"), ("steak", "pork_chop"), ("ramen", "pho"),
    ("chocolate_cake", "chocolate_mousse"), ("beef_tartare", "tuna_tartare"), ("ice_cream", "frozen_yogurt"),
]
SPLIT_ORDER = ["train", "val", "test"]


def crop(root, entry, size=224):
    with Image.open(image_path(root, entry)) as im:
        return np.asarray(T.CenterCrop(size)(T.Resize(256)(im.convert("RGB"))))


def describe(x: pd.Series) -> dict:
    q = x.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            **{f"p{int(k * 100)}": float(v) for k, v in q.items()}}


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/food101/_base.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default="outputs/eda/food101")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = apply_overrides(load_config(resolve_path(args.config)), args.set)
    dcfg = cfg["data"]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    root = dcfg["image_root"]

    split, lists = load_split(dcfg["split_dir"])
    classes = split["classes"]
    idx = load_index(dcfg["index_file"])
    which = {e: s for s in SPLIT_ORDER for e in lists[s]}
    idx["split"] = idx["entry"].map(which)
    assert idx["split"].notna().all(), "index and split lists disagree"
    summary = {"dataset": "food101", "n_classes": len(classes), "n_images": int(len(idx))}

    # ---- class balance
    counts = idx.groupby(["class", "split"]).size().unstack("split").reindex(index=classes, columns=SPLIT_ORDER)
    counts.to_csv(out / "class_counts.csv")
    summary["class_counts"] = {s: {"total": int(counts[s].sum()), "min": int(counts[s].min()),
                                   "max": int(counts[s].max()), "imbalance_ratio": float(counts[s].max() / counts[s].min())}
                               for s in SPLIT_ORDER}
    fig, ax = plt.subplots(figsize=(22, 4.5))
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for s in SPLIT_ORDER:
        ax.bar(x, counts[s].values, bottom=bottom, label=f"{s} (n={int(counts[s].sum()):,})", width=0.85)
        bottom += counts[s].values
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_ylabel("images")
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.legend(loc="lower right")
    ax.set_title("Images per class and split (stacked)")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---- image sizes (raw archive) and file sizes
    has_raw = "raw_w" in idx.columns and idx["raw_w"].notna().all()
    wcol, hcol, bcol = ("raw_w", "raw_h", "raw_bytes") if has_raw else ("w", "h", "bytes")
    idx["aspect"] = idx[wcol] / idx[hcol]
    idx["max_side"] = idx[[wcol, hcol]].max(axis=1)
    idx["min_side"] = idx[[wcol, hcol]].min(axis=1)
    summary["image_size_source"] = "raw archive" if has_raw else "pre-resized copy"
    summary["width"] = describe(idx[wcol])
    summary["height"] = describe(idx[hcol])
    summary["aspect_ratio"] = describe(idx["aspect"])
    summary["file_kb"] = describe(idx[bcol] / 1024)
    summary["share_max_side_512"] = float((idx["max_side"] == 512).mean())
    summary["share_min_side_below_224"] = float((idx["min_side"] < 224).mean())
    summary["share_portrait"] = float((idx["aspect"] < 0.98).mean())
    summary["share_square"] = float(idx["aspect"].between(0.98, 1.02).mean())
    summary["share_landscape"] = float((idx["aspect"] > 1.02).mean())
    if has_raw:
        summary["raw_modes"] = {k: int(v) for k, v in idx["raw_mode"].value_counts().items()}
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    axes[0].hist(idx[wcol], bins=60, color="tab:blue")
    axes[0].set_title("width (px)")
    axes[1].hist(idx[hcol], bins=60, color="tab:orange")
    axes[1].set_title("height (px)")
    axes[2].hist(idx["aspect"].clip(0.3, 3), bins=60, color="tab:green")
    axes[2].set_title("aspect ratio w/h (clipped 0.3-3)")
    axes[3].hist((idx[bcol] / 1024).clip(upper=idx[bcol].quantile(0.99) / 1024), bins=60, color="tab:purple")
    axes[3].set_title("file size (KB, clipped at p99)")
    for ax in axes:
        ax.set_ylabel("images")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Image geometry ({summary['image_size_source']}, all {len(idx):,} images)")
    fig.tight_layout()
    fig.savefig(out / "image_sizes.png", dpi=150)
    plt.close(fig)

    # ---- pixel statistics per split
    rows = []
    for s in SPLIT_ORDER:
        sub = idx[idx.split == s]
        mean, std = channel_stats(sub)
        rows.append({"split": s, "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
                     "std_r": std[0], "std_g": std[1], "std_b": std[2],
                     "brightness_mean": sub["brightness"].mean(), "contrast_mean": sub["contrast"].mean(),
                     "saturation_mean": sub["saturation"].mean(),
                     "share_low_saturation_lt_0.05": float((sub["saturation"] < 0.05).mean()),
                     "share_dark_lt_0.2": float((sub["brightness"] < 0.2).mean()),
                     "share_bright_gt_0.8": float((sub["brightness"] > 0.8).mean())})
    px = pd.DataFrame(rows)
    px.to_csv(out / "pixel_stats.csv", index=False)
    summary["pixel_stats"] = px.set_index("split").to_dict("index")
    summary["normalization_used_by_b0"] = split["normalization"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, col in zip(axes, ["brightness", "contrast", "saturation"]):
        for s in SPLIT_ORDER:
            ax.hist(idx.loc[idx.split == s, col], bins=60, density=True, histtype="step", lw=1.5, label=s)
        ax.set_title(f"per-image {col}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Pixel statistics per split (luma mean, luma std, mean max-min channel spread)")
    fig.tight_layout()
    fig.savefig(out / "pixel_stats.png", dpi=150)
    plt.close(fig)
    cb = idx.groupby("class")[["brightness", "saturation", "contrast"]].mean().sort_values("brightness")
    cb.to_csv(out / "class_brightness.csv")
    summary["darkest_classes"] = cb.head(5)["brightness"].round(3).to_dict()
    summary["brightest_classes"] = cb.tail(5)["brightness"].round(3).to_dict()

    # ---- sample grids
    by_class = {c: [] for c in classes}
    for e in lists["train"]:
        by_class[e.split("/", 1)[0]].append(e)
    first = [rng.choice(by_class[c]) for c in classes]
    plot_image_grid([crop(root, e) for e in first], [c.replace("_", " ") for c in classes],
                    out / "samples_all_classes.png", ncols=11, cell=1.6, fontsize=6,
                    suptitle="One random training image per class (centre crop as seen at eval time)")
    imgs, titles = [], []
    for a, b in [pr for pr in CONFUSABLE_PAIRS if set(pr) <= set(classes)]:
        for c in (a, b):
            for e in rng.choice(by_class[c], 5, replace=False):
                imgs.append(crop(root, e))
                titles.append(c)
    if imgs:
        plot_image_grid(imgs, titles, out / "confusable_pairs.png", ncols=10, cell=1.8, fontsize=7,
                        suptitle="Visually similar class pairs (5 train images per class, pair per row)")

    # ---- near-duplicates (pHash)
    h = {s: hex_to_uint64(idx.set_index("entry").loc[lists[s], "phash"]) for s in SPLIT_ORDER}
    thr = int(split["near_duplicates"]["max_hamming"])
    pair_rows = []
    for qa, ra in [("train", "train"), ("val", "train"), ("val", "val"), ("test", "train"), ("test", "val"),
                   ("test", "test")]:
        i, j, d = near_duplicate_pairs(h[qa], h[ra], thr, same_set=qa == ra)
        for a, b, dd in zip(i, j, d):
            ea, eb = lists[qa][a], lists[ra][b]
            pair_rows.append({"split_a": qa, "entry_a": ea, "split_b": ra, "entry_b": eb, "hamming": int(dd),
                              "same_class": ea.split("/")[0] == eb.split("/")[0]})
    nd = pd.DataFrame(pair_rows, columns=["split_a", "entry_a", "split_b", "entry_b", "hamming", "same_class"])
    nd.to_csv(out / "near_duplicates.csv", index=False)
    summary["near_duplicates"] = {
        "max_hamming": thr,
        "pairs": {f"{a}-{b}": int(len(g)) for (a, b), g in nd.groupby(["split_a", "split_b"])},
        "pairs_with_different_labels": int((~nd["same_class"]).sum()) if len(nd) else 0,
        "exact_hash_pairs": int((nd["hamming"] == 0).sum()) if len(nd) else 0,
        "val_moved_to_train_by_split": split["near_duplicates"]["n_val_moved_to_train"],
        "test_flagged": split["near_duplicates"]["n_test_flagged"],
        "test_flagged_share": split["near_duplicates"]["n_test_flagged"] / split["n_test"],
    }
    if len(nd):
        cross = nd[nd.split_a != nd.split_b]
        show = (cross if len(cross) else nd).sort_values("hamming").head(8)
        imgs, titles = [], []
        for _, r in show.iterrows():
            imgs += [crop(root, r.entry_a), crop(root, r.entry_b)]
            titles += [f"{r.split_a}: {r.entry_a}", f"{r.split_b}: {r.entry_b} (d={r.hamming})"]
        plot_image_grid(imgs, titles, out / "near_duplicates_examples.png", ncols=4, cell=2.4, fontsize=6,
                        suptitle=f"Near-duplicate pairs (pHash Hamming <= {thr}), lowest distance first")

    # ---- ImageNet overlap
    summary["imagenet_counterparts"] = FOOD101_TO_IMAGENET
    summary["split"] = {k: split[k] for k in ("split_seed", "val_fraction", "n_train", "n_val", "n_test",
                                               "fingerprints", "split_unit")}
    save_json(summary, out / "eda_summary.json")

    cc = summary["class_counts"]
    pxs = summary["pixel_stats"]
    nds = summary["near_duplicates"]
    md = ["# EDA summary - Food-101", "",
          f"{len(classes)} classes, {len(idx):,} images. Split seed {split['split_seed']}, "
          f"fingerprints {split['fingerprints']}.", "",
          "## Class balance", "",
          md_table(["Split", "Images", "Min / class", "Max / class", "Max/min"],
                   [[s, f"{cc[s]['total']:,}", cc[s]["min"], cc[s]["max"], f"{cc[s]['imbalance_ratio']:.3f}"]
                    for s in SPLIT_ORDER]), "",
          f"## Image geometry ({summary['image_size_source']})", "",
          md_table(["Quantity", "Mean", "Min", "p5", "Median", "p95", "Max"],
                   [[n, f"{summary[k]['mean']:.1f}", f"{summary[k]['min']:.0f}" if k != "aspect_ratio" else f"{summary[k]['min']:.2f}",
                     f"{summary[k]['p5']:.2f}", f"{summary[k]['p50']:.2f}", f"{summary[k]['p95']:.2f}",
                     f"{summary[k]['max']:.2f}"]
                    for n, k in [("width (px)", "width"), ("height (px)", "height"),
                                 ("aspect w/h", "aspect_ratio"), ("file size (KB)", "file_kb")]]), "",
          f"- max side = 512 px: {100 * summary['share_max_side_512']:.1f} % of images",
          f"- shorter side < 224 px: {100 * summary['share_min_side_below_224']:.2f} %",
          f"- landscape / square / portrait: {100 * summary['share_landscape']:.1f} / "
          f"{100 * summary['share_square']:.1f} / {100 * summary['share_portrait']:.1f} %",
          f"- raw colour modes: {summary.get('raw_modes', 'n/a')}", "",
          "## Pixel statistics", "",
          md_table(["Split", "Mean RGB", "Std RGB", "Brightness", "Saturation", "Low-sat. (<0.05)"],
                   [[s, " / ".join(f"{pxs[s][f'mean_{c}']:.3f}" for c in "rgb"),
                     " / ".join(f"{pxs[s][f'std_{c}']:.3f}" for c in "rgb"),
                     f"{pxs[s]['brightness_mean']:.3f}", f"{pxs[s]['saturation_mean']:.3f}",
                     f"{100 * pxs[s]['share_low_saturation_lt_0.05']:.2f} %"] for s in SPLIT_ORDER]), "",
          f"Darkest classes: {summary['darkest_classes']}", "",
          f"Brightest classes: {summary['brightest_classes']}", "",
          f"## Near-duplicates (pHash, Hamming <= {thr})", "",
          md_table(["Pair type", "Pairs"], [[k, v] for k, v in nds["pairs"].items()]), "",
          f"- pairs whose two images carry different labels: {nds['pairs_with_different_labels']}",
          f"- val images moved to train by make_splits: {nds['val_moved_to_train_by_split']}",
          f"- test images flagged (near-duplicate of train/val): {nds['test_flagged']} "
          f"({100 * nds['test_flagged_share']:.2f} % of test)", ""]
    (out / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"EDA written to {out}")


if __name__ == "__main__":
    main()
