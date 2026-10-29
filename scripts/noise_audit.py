#!/usr/bin/env python
"""Manual label-noise audit of the train split (proposal section 2.4).

Step 1 - make the sheets (20 random classes x 10 random train images, seed 42):
  python scripts/noise_audit.py make
  -> outputs/eda/food101/noise_audit/sheet_XX_<class>.png   (numbered tiles 0-9)
  -> data/noise_audit/label_audit.csv                       (commit after filling in)
Step 2 - a person looks at every sheet and fills the `verdict` column of the CSV:
  ok      the image shows the labelled dish
  wrong   the image shows a different dish / no food at all
  unsure  cannot tell (counted separately, excluded from the rate)
  `note` is free text (e.g. "menu photo", "shows pho, not ramen").
Step 3 - score:
  python scripts/noise_audit.py score
  -> outputs/eda/food101/noise_audit/audit_result.json / .md   (wrong-label rate + Wilson 95 % CI)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from torchvision import transforms as T

from src.data.food101 import image_path, load_split
from src.utils.config import load_config
from src.utils.io import resolve_path, save_json
from src.viz.plots import plot_image_grid

VERDICTS = {"ok", "wrong", "unsure"}


def wilson(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def make(args, cfg):
    split, lists = load_split(cfg["data"]["split_dir"])
    classes = split["classes"]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(classes, args.n_classes, replace=False).tolist())
    csv = resolve_path(args.csv)
    if csv.exists() and not args.force:
        print(f"{csv} already exists (use --force to overwrite - it may contain your verdicts)")
        return
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tf = T.Compose([T.Resize(256), T.CenterCrop(256)])
    rows = []
    for k, c in enumerate(chosen):
        pool = sorted(e for e in lists["train"] if e.startswith(c + "/"))
        picks = rng.choice(pool, args.per_class, replace=False)
        imgs = []
        for t, e in enumerate(picks):
            with Image.open(image_path(cfg["data"]["image_root"], e)) as im:
                imgs.append(np.asarray(tf(im.convert("RGB"))))
            rows.append({"sheet": k, "tile": t, "class": c, "entry": e, "verdict": "", "note": ""})
        plot_image_grid(imgs, [f"#{t}  {e.split('/')[1]}" for t, e in enumerate(picks)],
                        out / f"sheet_{k:02d}_{c}.png", ncols=5, cell=3.0, fontsize=9,
                        suptitle=f"Sheet {k:02d} - label: {c.replace('_', ' ')}  (does each tile show this dish?)")
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{len(rows)} images on {len(chosen)} sheets -> {out}\nFill the 'verdict' column of {csv} "
          f"with ok / wrong / unsure, then run: python scripts/noise_audit.py score")


def score(args, cfg):
    df = pd.read_csv(resolve_path(args.csv), keep_default_na=False)
    df["verdict"] = df["verdict"].str.strip().str.lower()
    bad = df[~df["verdict"].isin(VERDICTS)]
    if len(bad):
        print(f"{len(bad)} rows have no valid verdict (expected one of {sorted(VERDICTS)}), e.g.:")
        print(bad.head(10).to_string(index=False))
        sys.exit(1)
    judged = df[df.verdict != "unsure"]
    k, n = int((judged.verdict == "wrong").sum()), int(len(judged))
    lo, hi = wilson(k, n)
    per_class = df.groupby("class")["verdict"].value_counts().unstack(fill_value=0)
    result = {
        "n_images": int(len(df)), "n_classes": int(df["class"].nunique()),
        "n_unsure": int((df.verdict == "unsure").sum()), "n_judged": n, "n_wrong": k,
        "wrong_label_rate": k / n if n else None, "wilson_95ci": [lo, hi],
        "per_class": per_class.to_dict("index"),
        "wrong_examples": df[df.verdict == "wrong"][["class", "entry", "note"]].to_dict("records"),
        "protocol": f"{df['class'].nunique()} random classes x {df.groupby('class').size().iloc[0]} random "
                    f"train images (seed {args.seed}); 'unsure' excluded from the rate",
    }
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "audit_result.json")
    lines = ["# Label-noise audit (train split)", "", result["protocol"], "",
             f"- judged images: {n} (unsure: {result['n_unsure']})",
             f"- wrong labels: {k} -> rate {100 * k / max(n, 1):.1f} % (Wilson 95 % CI "
             f"{100 * lo:.1f}-{100 * hi:.1f} %)", "", "## Wrong-label examples", ""]
    lines += [f"- `{r['entry']}` ({r['class']}): {r['note']}" for r in result["wrong_examples"]] or ["- none"]
    (out / "audit_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["make", "score"])
    p.add_argument("--config", default="configs/food101/_base.yaml")
    p.add_argument("--csv", default="data/noise_audit/label_audit.csv")
    p.add_argument("--out", default="outputs/eda/food101/noise_audit")
    p.add_argument("--n-classes", type=int, default=20)
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(resolve_path(args.config))
    (make if args.action == "make" else score)(args, cfg)


if __name__ == "__main__":
    main()
