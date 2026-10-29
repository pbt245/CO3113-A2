#!/usr/bin/env python
"""Aggregate all finished Food-101 runs into report-ready tables and figures.

Usage:  python scripts/compare.py
Outputs (outputs/comparison/food101/):
  runs.csv                  one row per run (traceability: run_id, commit, split fingerprint, checkpoint hash)
  summary.csv / summary.md  mean ± std over seeds (main models + ablations)
  decisions.md              decision rule of the proposal applied to every model pair / ablation
  main_curves.png           train loss / val loss / val macro-F1 per model (lowest seed)
  f1_vs_gmacs.png, f1_vs_latency.png, f1_vs_train_time.png
  per_class_f1.csv/.png     mean per-class F1 per experiment
  imagenet_overlap.md       macro-F1 on classes with / without an ImageNet-1k counterpart
  mcnemar.csv               paired tests (shared seed) between main models and ablation vs reference
  data_efficiency.md        E3: macro-F1 ratio vs the full-data reference
  top_confusions_<exp>.csv  most frequent confusions of each main model (lowest seed)
"""
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine.metrics import mcnemar_test
from src.utils.config import load_config
from src.utils.io import load_json, resolve_path
from src.viz.plots import plot_heatmap

MAIN_ORDER = ["b0_simple_cnn", "p1_resnet50", "p2_efficientnet_b0", "p3_convnext_tiny"]
METRICS = ["test_accuracy", "test_macro_f1", "test_top5", "test_ece", "test_macro_f1_no_dups", "test_loss",
           "params", "gmacs", "best_epoch", "epochs_run", "train_time_s", "train_time_per_epoch_s",
           "train_img_per_s", "throughput_img_s", "latency_ms_bs1", "f1_imagenet", "f1_no_imagenet"]


def order_key(name):
    return (MAIN_ORDER.index(name) if name in MAIN_ORDER else len(MAIN_ORDER), name)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pm(mean, std, scale=1.0, digits=2):
    if pd.isna(mean):
        return "n/a"
    if pd.isna(std):
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def collect_runs(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(dataset_dir.glob("*/seed*/eval/test_metrics.json")):
        run_dir = mpath.parent.parent
        if not (run_dir / "train_summary.json").exists():
            continue
        m = load_json(mpath)
        s = load_json(run_dir / "train_summary.json")
        cfg = load_config(run_dir / "config.yaml")
        env = load_json(run_dir / "env.json") if (run_dir / "env.json").exists() else {}
        exp = cfg["experiment"]
        if exp.get("group") == "debug":
            continue
        git = env.get("git") or {}
        gpus = env.get("gpus") or []
        try:
            rel = str(run_dir.relative_to(ROOT).as_posix())
        except ValueError:
            rel = str(run_dir)
        nd = m.get("without_near_duplicates") or {}
        rows.append({
            "experiment": exp["name"], "id": exp.get("id"), "group": exp.get("group", "main"),
            "mandatory": exp.get("mandatory", True), "reference": exp.get("reference"),
            "changed_factor": exp.get("changed_factor"), "hypothesis": exp.get("hypothesis"),
            "model": cfg["model"]["name"], "seed": int(cfg["seed"]),
            "train_fraction": float(cfg["data"].get("train_fraction", 1.0)),
            "params": int(m["parameters"]["total"]), "params_trainable": int(m["parameters"]["trainable"]),
            "gmacs": m.get("gmacs_per_image"),
            "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"], "test_top5": m["top5_accuracy"],
            "test_ece": m["ece"], "test_loss": m["loss"], "test_macro_f1_no_dups": nd.get("macro_f1"),
            "n_test_excluded_dups": nd.get("n_excluded"),
            "f1_imagenet": m.get("macro_f1_imagenet_counterpart"), "f1_no_imagenet": m.get("macro_f1_no_imagenet_counterpart"),
            "acc_ci_low": m["bootstrap_ci"]["accuracy"][0], "acc_ci_high": m["bootstrap_ci"]["accuracy"][1],
            "f1_ci_low": m["bootstrap_ci"]["macro_f1"][0], "f1_ci_high": m["bootstrap_ci"]["macro_f1"][1],
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
            "best_val_macro_f1": s["best_val_macro_f1"],
            "train_time_s": s["train_time_s"], "train_time_per_epoch_s": s["train_time_per_epoch_s"],
            "train_img_per_s": s.get("train_img_per_s_mean"),
            "throughput_img_s": m["inference"]["throughput_img_per_s"],
            "latency_ms_bs1": m["inference"]["latency_ms_bs1_mean"], "inference_amp": m["inference"]["amp"],
            "device": s.get("device"), "gpu": gpus[0]["name"] if gpus else None, "amp": s.get("amp"),
            "run_id": cfg.get("run", {}).get("run_id"), "commit": git.get("commit"), "dirty": git.get("dirty"),
            "split_fingerprint_train": (m.get("split_fingerprints") or {}).get("train"),
            "checkpoint_epoch": m["checkpoint"]["epoch"], "checkpoint_sha256": m["checkpoint"]["sha256"],
            "run_dir": rel, "_run_dir": run_dir,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("experiment")
    agg = g[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = g.size()
    agg["seeds"] = g["seed"].apply(lambda s: " ".join(map(str, sorted(s))))
    meta = g[["id", "group", "mandatory", "model", "reference", "changed_factor", "hypothesis",
              "train_fraction", "gpu"]].first()
    out = meta.join(agg)
    order = sorted(out.index, key=order_key)
    return out.loc[order].reset_index()


def run_dir_of(df, exp, seed):
    return df[(df.experiment == exp) & (df.seed == seed)]["_run_dir"].iloc[0]


SUMMARY_HEADERS = ["ID", "Experiment", "Seeds", "Params (M)", "GMACs", "Test acc (%)", "Test macro-F1 (%)",
                   "Top-5 (%)", "ECE", "Macro-F1 w/o dups (%)", "Best epoch", "Train min/epoch",
                   "Train img/s", "Infer img/s (bs128)", "Latency bs1 (ms)"]


def summary_rows(agg):
    return [[r["id"] or "", r["experiment"] + ("" if r["mandatory"] else " *"), r["seeds"],
             f"{r['params_mean'] / 1e6:.2f}", f"{r['gmacs_mean']:.2f}",
             pm(r["test_accuracy_mean"], r["test_accuracy_std"], 100),
             pm(r["test_macro_f1_mean"], r["test_macro_f1_std"], 100),
             pm(r["test_top5_mean"], r["test_top5_std"], 100),
             pm(r["test_ece_mean"], r["test_ece_std"], 1, 3),
             pm(r["test_macro_f1_no_dups_mean"], r["test_macro_f1_no_dups_std"], 100),
             pm(r["best_epoch_mean"], r["best_epoch_std"], 1, 1),
             pm(r["train_time_per_epoch_s_mean"] / 60, r["train_time_per_epoch_s_std"] / 60, 1, 1),
             pm(r["train_img_per_s_mean"], r["train_img_per_s_std"], 1, 0),
             pm(r["throughput_img_s_mean"], r["throughput_img_s_std"], 1, 0),
             pm(r["latency_ms_bs1_mean"], r["latency_ms_bs1_std"], 1, 2)] for _, r in agg.iterrows()]


def plot_curves(df, experiments, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for exp in experiments:
        seed = sorted(df[df.experiment == exp].seed)[0]
        h = pd.read_csv(run_dir_of(df, exp, seed) / "history.csv")
        axes[0].plot(h.epoch, h.train_loss, label=f"{exp} (s{seed})")
        axes[1].plot(h.epoch, h.val_loss, label=f"{exp} (s{seed})")
        axes[2].plot(h.epoch, h.val_macro_f1, label=f"{exp} (s{seed})")
    for ax, t in zip(axes, ["Train loss (augmented, label smoothing if used)", "Validation loss",
                            "Validation macro-F1"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(agg, xcol, path, xlabel, title, logx=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in agg.iterrows():
        marker = "o" if r["group"] == "main" else "x"
        ax.errorbar(r[f"{xcol}_mean"], 100 * r["test_macro_f1_mean"],
                    yerr=0 if pd.isna(r["test_macro_f1_std"]) else 100 * r["test_macro_f1_std"],
                    fmt=marker, capsize=3)
        ax.annotate(r["experiment"], (r[f"{xcol}_mean"], 100 * r["test_macro_f1_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test macro-F1 (%)  mean ± std over seeds")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_f1(df, experiments):
    table, classes = {}, None
    for exp in experiments:
        frames = []
        for rd in df[df.experiment == exp]["_run_dir"]:
            rep = pd.read_csv(rd / "eval" / "classification_report.csv")
            classes = rep["class"].tolist()
            frames.append(rep["f1"].values)
        table[exp] = np.mean(frames, axis=0)
    return pd.DataFrame(table, index=classes)


def mcnemar_pair(df, a, b):
    common = sorted(set(df[df.experiment == a].seed) & set(df[df.experiment == b].seed))
    if not common:
        return None
    s = 42 if 42 in common else common[0]
    pa = pd.read_csv(run_dir_of(df, a, s) / "eval" / "predictions.csv")
    pb = pd.read_csv(run_dir_of(df, b, s) / "eval" / "predictions.csv")
    if not np.array_equal(pa.y_true.values, pb.y_true.values):
        print(f"warning: test labels differ between {a} and {b}; skipping McNemar")
        return None
    r = mcnemar_test(pa.y_true.values, pa.y_pred.values, pb.y_pred.values)
    return {"model_a": a, "model_b": b, "seed": s,
            "acc_a": float((pa.y_true == pa.y_pred).mean()), "acc_b": float((pb.y_true == pb.y_pred).mean()),
            "a_right_b_wrong": r["b"], "a_wrong_b_right": r["c"],
            "statistic": r["statistic"], "p_value": r["p_value"], "method": r["method"]}


def decide(agg, a, b, mc):
    """Proposal section 4: A better than B only if (mean gap > max seed std) AND McNemar p < 0.05."""
    A, B = agg[agg.experiment == a].iloc[0], agg[agg.experiment == b].iloc[0]
    gap = A["test_macro_f1_mean"] - B["test_macro_f1_mean"]
    stds = [A["test_macro_f1_std"], B["test_macro_f1_std"]]
    enough_seeds = A["n_seeds"] >= 2 and B["n_seeds"] >= 2
    max_std = max(stds) if enough_seeds else float("nan")
    p = mc["p_value"] if mc else float("nan")
    cond1 = bool(enough_seeds and abs(gap) > max_std)
    cond2 = bool(mc is not None and p < 0.05)
    winner = a if gap > 0 else b
    if not enough_seeds:
        verdict = f"preliminary: needs >= 2 seeds per arm (has {int(A['n_seeds'])}/{int(B['n_seeds'])})"
    elif cond1 and cond2:
        verdict = f"**{winner} better**"
    else:
        verdict = "no decision (difference not established)"
    return [a, b, f"{100 * gap:+.2f}", "n/a" if np.isnan(max_std) else f"{100 * max_std:.2f}",
            "n/a" if np.isnan(p) else f"{p:.3g}", "yes" if cond1 else "no", "yes" if cond2 else "no", verdict]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="food101")
    p.add_argument("--runs-root", default="outputs/runs")
    p.add_argument("--out", default="outputs/comparison")
    args = p.parse_args()

    dataset_dir = resolve_path(args.runs_root) / args.dataset
    out = resolve_path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    df = collect_runs(dataset_dir)
    if df.empty:
        print(f"No finished runs under {dataset_dir}")
        return
    df.drop(columns=["_run_dir"]).to_csv(out / "runs.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out / "summary.csv", index=False)
    main_exps = list(agg[agg.group == "main"]["experiment"])
    abl = agg[agg.group == "ablation"]
    gpus = sorted(set(g for g in df.gpu if g))

    md = [f"# Comparison - {args.dataset}", "",
          "Mean ± sample std over training seeds (same split for every run). Checkpoint = best validation "
          "macro-F1; the test set is evaluated once per checkpoint. `*` = optional experiment. "
          "\"w/o dups\" excludes the test images flagged as pHash near-duplicates of train/val. "
          f"Hardware: {gpus or sorted(set(df.device))}; inference timing fp16 on CUDA = {sorted(set(df.inference_amp))}.",
          "", "## Main models", "", md_table(SUMMARY_HEADERS, summary_rows(agg[agg.group == "main"]))]
    if not abl.empty:
        md += ["", "## Ablations / controlled experiments", "", md_table(SUMMARY_HEADERS, summary_rows(abl))]
    dirty = df[df.dirty == True]
    md += ["", "## Traceability", "",
           f"Commits used: {sorted(set(c for c in df.commit if c))}",
           f"Runs with uncommitted changes: {len(dirty)}",
           f"Train split fingerprints: {sorted(set(f for f in df.split_fingerprint_train if f))}"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if main_exps:
        plot_curves(df, main_exps, out / "main_curves.png", f"{args.dataset}: main models")
        main_agg = agg[agg.group == "main"]
        scatter(main_agg, "gmacs", out / "f1_vs_gmacs.png", "GMACs per 224 px image (log)", "Macro-F1 vs compute")
        scatter(main_agg, "latency_ms_bs1", out / "f1_vs_latency.png",
                "latency per image, batch size 1 (ms, log)", "Macro-F1 vs inference latency")
        scatter(main_agg, "train_time_s", out / "f1_vs_train_time.png",
                "total training time (s, log)", "Macro-F1 vs training cost")
        for exp in main_exps:
            s = sorted(df[df.experiment == exp].seed)[0]
            pd.read_csv(run_dir_of(df, exp, s) / "eval" / "top_confusions.csv").to_csv(
                out / f"top_confusions_{exp}.csv", index=False)

    pcf = per_class_f1(df, list(agg["experiment"]))
    pcf.to_csv(out / "per_class_f1.csv")
    if main_exps:
        best = agg[agg.group == "main"].sort_values("test_macro_f1_mean").iloc[-1]["experiment"]
        sorted_pcf = pcf[main_exps].sort_values(best)
        plot_heatmap(100 * sorted_pcf.values, list(sorted_pcf.index), main_exps, out / "per_class_f1.png",
                     title=f"Per-class test F1 (%), mean over seeds, sorted by {best}", fmt=".0f",
                     vmin=0, vmax=100, figsize=(2.2 * len(main_exps) + 3, 26))

    ov = [[r["experiment"], pm(r["f1_imagenet_mean"], r["f1_imagenet_std"], 100),
           pm(r["f1_no_imagenet_mean"], r["f1_no_imagenet_std"], 100),
           f"{100 * (r['f1_imagenet_mean'] - r['f1_no_imagenet_mean']):+.2f}"] for _, r in agg.iterrows()]
    (out / "imagenet_overlap.md").write_text(
        "# Macro-F1 on classes with / without an ImageNet-1k counterpart\n\n"
        "Classes with a counterpart: see src/data/imagenet_overlap.py (7 of 101).\n\n"
        + md_table(["Experiment", "With counterpart (%)", "Without (%)", "Gap (pp)"], ov) + "\n", encoding="utf-8")

    # McNemar + decision rule: every pair of main models, every ablation vs its reference
    pairs = list(itertools.combinations(main_exps, 2))
    pairs += [(r["experiment"], r["reference"]) for _, r in abl.iterrows() if r["reference"] in set(agg.experiment)]
    mc_rows, dec_rows = [], []
    for a, b in pairs:
        mc = mcnemar_pair(df, a, b)
        if mc:
            mc_rows.append(mc)
        dec_rows.append(decide(agg, a, b, mc))
    if mc_rows:
        pd.DataFrame(mc_rows).to_csv(out / "mcnemar.csv", index=False)
    hyp = [f"- **{r['id']} {r['experiment']}** (vs {r['reference']}; changed: {r['changed_factor']}): {r['hypothesis']}"
           for _, r in abl.iterrows()]
    (out / "decisions.md").write_text(
        "# Decisions (proposal section 4 rule)\n\nA is declared better than B only if |mean test macro-F1 gap| "
        "over seeds exceeds the larger seed std **and** McNemar on seed 42 gives p < 0.05.\n\n"
        + md_table(["A", "B", "Δ macro-F1 A−B (pp)", "max std (pp)", "McNemar p", "gap > std", "p < 0.05",
                    "Verdict"], dec_rows)
        + ("\n\n## Pre-registered hypotheses\n\n" + "\n".join(hyp) if hyp else "") + "\n", encoding="utf-8")

    # E3 data efficiency
    e3 = agg[agg.experiment.str.startswith("e3_")]
    if not e3.empty:
        rows = []
        for _, r in e3.sort_values(["reference", "train_fraction"]).iterrows():
            ref = agg[agg.experiment == r["reference"]]
            if ref.empty:
                continue
            ref_f1 = ref.iloc[0]["test_macro_f1_mean"]
            rows.append([r["reference"], f"{100 * r['train_fraction']:.0f} %", f"{100 * r['test_macro_f1_mean']:.2f}",
                         f"{100 * ref_f1:.2f}", f"{r['test_macro_f1_mean'] / ref_f1:.3f}"])
        (out / "data_efficiency.md").write_text(
            "# E3 - data efficiency\n\n" + md_table(["Model", "Train fraction", "Macro-F1 (%)",
                                                     "Full-data macro-F1 (%)", "Ratio"], rows) + "\n", encoding="utf-8")

    print((out / "summary.md").read_text(encoding="utf-8"))
    print((out / "decisions.md").read_text(encoding="utf-8"))
    print(f"\nComparison written to {out}")


if __name__ == "__main__":
    main()
