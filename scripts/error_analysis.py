#!/usr/bin/env python
"""M3 error analysis of the main models on the test set (one seed per model, default 42).

  python scripts/error_analysis.py taxonomy        # automatic taxonomy, overlap, calibration figures
  python scripts/error_analysis.py review-make     # sheets + CSV for the manual review of P1 errors
  python scripts/error_analysis.py review-score    # score the filled-in review CSV

Automatic taxonomy - every test error of a model falls into exactly one bucket (first match wins):
  consensus   every pretrained main model predicts the same wrong class -> label problem or a
              genuinely ambiguous photo (no single model is to blame)
  sibling     true and predicted class are in the same culinary group (src/data/food_groups.py)
  near_miss   different group, but the true class is in the model's top 5
  gross       different group and the true class is not in the top 5
Orthogonal flags: confident (confidence > 0.9), near-duplicate of a train/val image.

Manual review (review-make / review-score), P1 errors only:
  stratum random    75 errors sampled uniformly (seed 42)       -> unbiased estimate over all errors
  stratum top_conf  the 50 most confident remaining errors      -> what the confident failures are
  verdict           label_wrong  the photo shows the predicted dish (or another dish), not the label
                    ambiguous    several dishes / dish not identifiable / both labels plausible
                    model_error  the photo clearly shows the labelled dish
  The reviewed sample (data/error_review/p1_error_review.csv) was drawn from the errors of the M2 P1 seed-42
  checkpoint (run food101-p1_resnet50-s42-20261029-195643, run folder in git at commit e39b497); its buckets are
  those of that checkpoint. review-score adds how many reviewed images the final (M3) seed-42 models still get
  wrong - label_wrong / ambiguous are properties of the image, so they should persist across models.
Outputs: outputs/error_analysis/food101/
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

from src.data.food101 import image_path, load_split
from src.data.food_groups import CLASS_TO_GROUP, FOOD_GROUPS
from src.engine.metrics import wilson_ci
from src.utils.config import load_config
from src.utils.io import load_json, resolve_path, save_json
from src.viz.plots import plot_heatmap, plot_image_grid

MAIN = {"b0_simple_cnn": "B0", "p1_resnet50": "P1", "p2_efficientnet_b0": "P2", "p3_convnext_tiny": "P3"}
PRETRAINED = ["p1_resnet50", "p2_efficientnet_b0", "p3_convnext_tiny"]
CALIBRATION = ["b0_simple_cnn", "p1_resnet50", "p2_efficientnet_b0", "p3_convnext_tiny",
               "e1_resnet50_linear_probe", "e2_resnet50_label_smoothing"]
BUCKETS = ["consensus", "sibling", "near_miss", "gross"]
VERDICTS = ["label_wrong", "ambiguous", "model_error"]
REVIEW_MODEL = "p1_resnet50"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def load_predictions(runs_root: Path, exps, seed: int) -> dict:
    preds = {}
    for exp in exps:
        f = runs_root / exp / f"seed{seed}" / "eval" / "predictions.csv"
        if f.exists():
            preds[exp] = pd.read_csv(f)
    return preds


def classify_errors(preds: dict, near_dups: set) -> dict:
    """Add bucket / flag columns to each model's prediction table (rows are aligned by test index)."""
    pre = [e for e in PRETRAINED if e in preds]
    consensus_pred = None
    if len(pre) >= 2:
        stack = np.stack([preds[e].y_pred.values for e in pre])
        y = preds[pre[0]].y_true.values
        agree = (stack == stack[0]).all(axis=0) & (stack[0] != y)
        consensus_pred = np.where(agree, stack[0], -1)
    out = {}
    for exp, df in preds.items():
        df = df.copy()
        top5 = df.top5.str.split().apply(lambda r: list(map(int, r)))
        in_top5 = np.array([t in r for t, r in zip(df.y_true, top5)])
        same_group = np.array([CLASS_TO_GROUP[a] == CLASS_TO_GROUP[b] for a, b in zip(df.true_name, df.pred_name)])
        bucket = np.where(same_group, "sibling", np.where(in_top5, "near_miss", "gross"))
        if consensus_pred is not None:
            bucket = np.where(df.y_pred.values == consensus_pred, "consensus", bucket)
        df["bucket"] = np.where(df.correct, "", bucket)
        df["confident"] = df.confidence > 0.9
        df["near_duplicate"] = df.entry.isin(near_dups)
        df["true_group"] = df.true_name.map(CLASS_TO_GROUP)
        df["pred_group"] = df.pred_name.map(CLASS_TO_GROUP)
        out[exp] = df
    return out


def taxonomy_table(tagged: dict):
    rows, records = [], []
    for exp, df in tagged.items():
        err = df[~df.correct]
        n = len(err)
        rec = {"experiment": exp, "n_test": len(df), "n_errors": n}
        cells = [MAIN.get(exp, exp), exp, f"{n:,} ({100 * n / len(df):.1f} %)"]
        for b in BUCKETS:
            k = int((err.bucket == b).sum())
            rec[b] = k
            cells.append(f"{k:,} ({100 * k / max(n, 1):.1f} %)")
        conf = err[err.confident]
        rec["confident"] = len(conf)
        rec["confident_consensus_or_sibling"] = int(conf.bucket.isin(["consensus", "sibling"]).sum())
        rec["near_duplicate_errors"] = int(err.near_duplicate.sum())
        cells += [f"{len(conf):,} ({100 * len(conf) / max(n, 1):.1f} %)",
                  f"{100 * rec['confident_consensus_or_sibling'] / max(len(conf), 1):.0f} %",
                  rec["near_duplicate_errors"]]
        rows.append(cells)
        records.append(rec)
    headers = ["ID", "Experiment", "Errors (rate)", "Consensus", "Sibling", "Near-miss (top-5)", "Gross",
               "Confident (> 0.9)", "Confident that are consensus/sibling", "Near-dup errors"]
    return md_table(headers, rows), pd.DataFrame(records)


def overlap_section(tagged: dict) -> str:
    exps = [e for e in MAIN if e in tagged]
    wrong = {e: ~tagged[e].correct.values for e in exps}
    n = len(next(iter(tagged.values())))
    lines = ["## Error overlap between models (same test images)", ""]
    rows = []
    for a in exps:
        for b in exps:
            if a < b:
                both = int((wrong[a] & wrong[b]).sum())
                rows.append([MAIN[a], MAIN[b], f"{wrong[a].sum():,}", f"{wrong[b].sum():,}", f"{both:,}",
                             f"{100 * both / max(min(wrong[a].sum(), wrong[b].sum()), 1):.0f} %"])
    lines.append(md_table(["A", "B", "Errors A", "Errors B", "Wrong in both", "Both / smaller error set"], rows))
    all_wrong = np.logical_and.reduce([wrong[e] for e in exps])
    any_right = ~all_wrong
    lines += ["", f"- Wrong for **all** {len(exps)} models ({', '.join(MAIN[e] for e in exps)}): "
                  f"{int(all_wrong.sum()):,} images ({100 * all_wrong.mean():.2f} %).",
              f"- Oracle (at least one model right): {100 * any_right.mean():.2f} % accuracy - the upper bound "
              f"of any selection/ensemble among these models."]
    pre = [e for e in PRETRAINED if e in tagged]
    if len(pre) >= 2:
        cons = (tagged[pre[0]].bucket == "consensus").values
        lines.append(f"- All pretrained models ({', '.join(MAIN[e] for e in pre)}) agree on the same wrong class: "
                     f"{int(cons.sum()):,} images ({100 * cons.sum() / n:.2f} % of the test set).")
    return "\n".join(lines)


def group_confusion(df: pd.DataFrame, path: Path, title: str):
    groups = list(FOOD_GROUPS)
    gi = {g: i for i, g in enumerate(groups)}
    cm = np.zeros((len(groups), len(groups)))
    err = df[~df.correct]
    for t, p in zip(err.true_group, err.pred_group):
        cm[gi[t], gi[p]] += 1
    share = 100 * cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plot_heatmap(share, groups, groups, path, title=title, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                 figsize=(10, 8))
    return float(np.trace(cm) / max(cm.sum(), 1))


def calibration_plots(runs_root: Path, seed: int, out: Path):
    runs = [(e, load_json(f)) for e in CALIBRATION
            if (f := runs_root / e / f"seed{seed}" / "eval" / "test_metrics.json").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for exp, m in runs:
        b = pd.DataFrame(m["calibration_bins"])
        b = b[b["count"] > 0]
        ax.plot(b.confidence, b.accuracy, marker="o", ms=3, label=f"{exp} (ECE {m['ece']:.3f})")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram, test set, seed {seed}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for exp, _ in runs:
        if exp not in MAIN:
            continue
        p = pd.read_csv(runs_root / exp / f"seed{seed}" / "eval" / "predictions.csv")
        ax.hist(p.confidence[~p.correct], bins=40, range=(0, 1), histtype="step", lw=1.5,
                label=f"{exp} errors (n={int((~p.correct).sum())})")
    ax.set_xlabel("confidence of the predicted class")
    ax.set_ylabel("number of wrong predictions")
    ax.set_title("Confidence of wrong predictions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)


def worst_classes(tagged: dict, k: int = 15) -> str:
    best = max(tagged, key=lambda e: tagged[e].correct.mean())
    df = tagged[best]
    rows = []
    recall = df.groupby("true_name").correct.mean().sort_values()
    for c in recall.index[:k]:
        err = df[(df.true_name == c) & ~df.correct]
        top = err.pred_name.value_counts().head(2)
        sib = (err.bucket.isin(["sibling", "consensus"])).mean() if len(err) else 0
        rows.append([c, CLASS_TO_GROUP[c], f"{100 * recall[c]:.1f}",
                     ", ".join(f"{p} ({n})" for p, n in top.items()), f"{100 * sib:.0f} %"])
    return (f"## {k} classes with the lowest recall ({best})\n\n"
            + md_table(["Class", "Group", "Recall (%)", "Most frequent wrong predictions",
                        "Errors that are sibling/consensus"], rows))


def taxonomy(args):
    runs_root = resolve_path(args.runs_root) / "food101"
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/food101/_base.yaml")
    _, lists = load_split(cfg["data"]["split_dir"])
    preds = load_predictions(runs_root, list(MAIN), args.seed)
    if not preds:
        print("No predictions found.")
        return
    tagged = classify_errors(preds, set(lists["test_near_duplicates"]))
    for exp, df in tagged.items():
        df.to_csv(out / f"errors_{exp}_seed{args.seed}.csv", index=False)
    table, records = taxonomy_table(tagged)
    records.to_csv(out / "taxonomy.csv", index=False)
    within = {}
    for exp, df in tagged.items():
        within[exp] = group_confusion(df, out / f"group_confusion_{exp}.png",
                                      f"{exp} seed {args.seed}: where errors go, by culinary group "
                                      f"(% of the row's errors)")
    calibration_plots(runs_root, args.seed, out)
    pre = [MAIN[e] for e in PRETRAINED if e in tagged]
    md = ["# Error taxonomy - food101 test set", "",
          f"Seed {args.seed} for every model. Buckets are exclusive, assigned in this order: **consensus** "
          f"(all pretrained models {pre} predict the same wrong class), **sibling** (true and predicted class "
          "in the same culinary group of `src/data/food_groups.py`), **near-miss** (other group, true class in "
          "top 5), **gross** (other group, not in top 5). Percentages are of the model's errors.", "",
          table, "",
          "Share of errors that stay inside the true class's culinary group (diagonal of the group matrix): "
          + ", ".join(f"{MAIN[e]} {100 * v:.0f} %" for e, v in within.items()) + ".", "",
          overlap_section(tagged), "", worst_classes(tagged), ""]
    (out / "taxonomy.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten to {out}")


def review_make(args):
    out = resolve_path(args.out)
    csv = resolve_path(args.review_csv)
    if csv.exists() and not args.force:
        print(f"{csv} already exists (use --force to overwrite - it may contain verdicts)")
        return
    f = out / f"errors_{REVIEW_MODEL}_seed{args.seed}.csv"
    if not f.exists():
        print(f"{f} missing: run `taxonomy` first")
        return
    df = pd.read_csv(f)
    err = df[~df.correct]
    rng = np.random.default_rng(args.seed)
    rand = err.loc[np.sort(rng.choice(err.index.values, args.n_random, replace=False))].assign(stratum="random")
    rest = err.drop(rand.index).sort_values("confidence", ascending=False).head(args.n_top).assign(stratum="top_conf")
    picks = pd.concat([rand, rest]).reset_index(drop=True)
    cfg = load_config(ROOT / "configs/food101/_base.yaml")
    tf = T.Compose([T.Resize(256), T.CenterCrop(256)])
    sheets = out / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    per = args.per_sheet
    rows = []
    for s in range(int(np.ceil(len(picks) / per))):
        chunk = picks.iloc[s * per:(s + 1) * per]
        imgs, titles = [], []
        for t, (_, r) in enumerate(chunk.iterrows()):
            with Image.open(image_path(cfg["data"]["image_root"], r.entry)) as im:
                imgs.append(np.asarray(tf(im.convert("RGB"))))
            titles.append(f"#{s:02d}-{t:02d}  label: {r.true_name}\npred: {r.pred_name} ({r.confidence:.2f})")
            rows.append({"sheet": s, "tile": t, "stratum": r.stratum, "entry": r.entry, "true_name": r.true_name,
                         "pred_name": r.pred_name, "confidence": round(float(r.confidence), 4),
                         "bucket": r.bucket, "verdict": "", "note": "", "rater": ""})
        plot_image_grid(imgs, titles, sheets / f"review_{s:02d}.png", ncols=4, cell=3.4, fontsize=9,
                        suptitle=f"P1 error review sheet {s:02d}: does the photo show the LABEL, the PREDICTION, "
                                 f"or is it ambiguous?")
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{len(rows)} errors on {s + 1} sheets -> {sheets}\nFill 'verdict' ({' / '.join(VERDICTS)}), 'note' "
          f"and 'rater' in {csv}, then run: python scripts/error_analysis.py review-score")


def review_score(args):
    df = pd.read_csv(resolve_path(args.review_csv), keep_default_na=False)
    df["verdict"] = df.verdict.str.strip().str.lower()
    bad = df[~df.verdict.isin(VERDICTS)]
    if len(bad):
        print(f"{len(bad)} rows without a valid verdict ({VERDICTS}):\n{bad.head(10).to_string(index=False)}")
        sys.exit(1)
    runs_root = resolve_path(args.runs_root) / "food101"
    preds = {e: p.set_index("entry") for e, p in load_predictions(runs_root, PRETRAINED, args.seed).items()}
    for e, p in preds.items():
        df[f"wrong_{MAIN[e]}"] = ~df.entry.map(p.correct).astype(bool)
        df[f"same_{MAIN[e]}"] = df.entry.map(p.pred_name) == df.pred_name
    tags = [MAIN[e] for e in preds]
    if tags:
        df["wrong_all"] = df[[f"wrong_{t}" for t in tags]].all(axis=1)
        df["same_all"] = df[[f"same_{t}" for t in tags]].all(axis=1)
    result, lines = {"model": REVIEW_MODEL, "n_raters": len(set(df.rater) - {""}), "strata": {}}, []
    rows = []
    for stratum, g in df.groupby("stratum", sort=False):
        n = len(g)
        res = {"n": n}
        cells = [stratum, n]
        for v in VERDICTS:
            k = int((g.verdict == v).sum())
            lo, hi = wilson_ci(k, n)
            res[v] = {"k": k, "rate": k / n, "wilson_95ci": [lo, hi]}
            cells.append(f"{k} ({100 * k / n:.0f} %, CI {100 * lo:.0f}-{100 * hi:.0f})")
        result["strata"][stratum] = res
        rows.append(cells)
    xt = pd.crosstab(df.bucket, df.verdict).reindex(index=BUCKETS, columns=VERDICTS, fill_value=0)
    result["bucket_x_verdict"] = xt.to_dict("index")
    pers = []
    for v in VERDICTS:
        g = df[df.verdict == v]
        cells = [v, len(g)] + [f"{int(g[f'wrong_{t}'].sum())} ({100 * g[f'wrong_{t}'].mean():.0f} %)" for t in tags]
        if tags:
            cells += [f"{int(g.wrong_all.sum())} ({100 * g.wrong_all.mean():.0f} %)",
                      f"{int(g.same_all.sum())} ({100 * g.same_all.mean():.0f} %)"]
        pers.append(cells)
        result.setdefault("persistence", {})[v] = {c: int(g[c].sum()) for c in g.columns if c.startswith(("wrong_", "same_"))}
    out = resolve_path(args.out)
    save_json(result, out / "review_result.json")
    lines = ["# Manual review of P1 test errors", "",
             f"Errors of the M2 {REVIEW_MODEL} seed-42 checkpoint (run food101-p1_resnet50-s42-20261029-195643, commit "
             f"e39b497). One rater per image; Wilson 95 % intervals.", "",
             md_table(["Stratum", "n"] + VERDICTS, rows), "",
             "## Automatic bucket x manual verdict (both strata)", "",
             md_table(["Bucket"] + VERDICTS, [[b] + [int(xt.loc[b, v]) for v in VERDICTS] for b in BUCKETS]), "",
             "## Persistence in the final (M3) models, seed " + str(args.seed), "",
             "How many reviewed images each final model still gets wrong; 'all' = every pretrained model wrong, "
             "'all, same class' = every pretrained model predicts the class the M2 model predicted.", "",
             md_table(["Verdict", "n"] + [f"{t} wrong" for t in tags] + (["all wrong", "all, same class"] if tags else []),
                      pers), "",
             "## Label-wrong and ambiguous examples", ""]
    lines += [f"- `{r.entry}` ({r.stratum}, label {r.true_name}, pred {r.pred_name} {r.confidence:.2f}): "
              f"**{r.verdict}** - {r.note}" for r in df[df.verdict != "model_error"].itertuples()]
    (out / "review_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["taxonomy", "review-make", "review-score"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-root", default="outputs/runs")
    p.add_argument("--out", default="outputs/error_analysis/food101")
    p.add_argument("--review-csv", default="data/error_review/p1_error_review.csv")
    p.add_argument("--n-random", type=int, default=75)
    p.add_argument("--n-top", type=int, default=50)
    p.add_argument("--per-sheet", type=int, default=16)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    {"taxonomy": taxonomy, "review-make": review_make, "review-score": review_score}[args.action](args)


if __name__ == "__main__":
    main()
