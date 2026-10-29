"""Evaluate the selected checkpoint of a run on the test split and write every artifact
needed for the report: metrics (top-1, macro-F1, top-5, ECE), bootstrap CI, metrics without the
flagged near-duplicate test images, per-class report, confusion matrices, predictions, top
confusions, correct / hard-correct / incorrect example grids, inference cost."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.food101 import build_datasets
from src.data.imagenet_overlap import FOOD101_TO_IMAGENET
from src.engine.inference import measure_inference
from src.engine.loops import predict
from src.engine.metrics import (bootstrap_ci, classification_metrics, expected_calibration_error,
                                top_confusions, top_k_accuracy)
from src.models import build_model, count_macs, count_parameters
from src.utils.config import apply_overrides, load_config
from src.utils.env import git_info, synchronize
from src.utils.io import save_json, sha256_file
from src.utils.seed import set_seed
from src.viz.plots import plot_confusion_matrix, plot_image_grid


def load_run(run_dir, device, checkpoint: str = "best.pt", overrides=None):
    run_dir = Path(run_dir)
    cfg = apply_overrides(load_config(run_dir / "config.yaml"), overrides)
    datasets, meta = build_datasets(cfg)
    model_cfg = dict(cfg["model"])
    if model_cfg["name"] != "simple_cnn":
        model_cfg["pretrained"] = False           # weights come from the checkpoint
    model = build_model(model_cfg, meta["num_classes"])
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return cfg, model, ckpt, ckpt_path, datasets, meta


def _subset_metrics(y_true, y_pred, probs, k):
    m = classification_metrics(y_true, y_pred, k)
    return {"n_samples": int(len(y_true)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "top5_accuracy": top_k_accuracy(y_true, probs, 5)}


def evaluate_run(run_dir, device, split: str = "test", logger=None, overrides=None) -> dict:
    log = logger.info if logger is not None else print
    run_dir = Path(run_dir)
    cfg, model, ckpt, ckpt_path, datasets, meta = load_run(run_dir, device, overrides=overrides)
    seed = int(cfg["seed"])
    set_seed(seed)
    ecfg = cfg.get("eval", {})
    classes = meta["classes"]
    k = meta["num_classes"]
    dataset = datasets[split]
    out_dir = run_dir / ("eval" if split == "test" else f"eval_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(ecfg.get("amp", False)) and device.type == "cuda"

    loader = DataLoader(dataset, batch_size=int(cfg["train"].get("eval_batch_size", 128)), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)), pin_memory=device.type == "cuda")
    synchronize(device)
    t0 = time.perf_counter()
    res = predict(model, loader, device, nn.CrossEntropyLoss(), use_amp=use_amp)
    synchronize(device)
    eval_pass_time = time.perf_counter() - t0

    y_true, y_pred, probs = res["y_true"], res["y_pred"], res["probs"]
    m = classification_metrics(y_true, y_pred, k)
    top5 = top_k_accuracy(y_true, probs, 5)
    ece = expected_calibration_error(y_true, probs, n_bins=int(ecfg.get("ece_bins", 15)))
    ci = bootstrap_ci(y_true, y_pred, k, n_boot=int(ecfg.get("bootstrap_samples", 1000)), seed=seed)
    cm = m["confusion_matrix"]

    # ---- leakage check: metrics without the test images flagged as near-duplicates of train/val
    without_dups = None
    if split == "test":
        flagged = set(meta["test_near_duplicates"])
        keep = np.array([e not in flagged for e in dataset.entries])
        without_dups = {**_subset_metrics(y_true[keep], y_pred[keep], probs[keep], k),
                        "n_excluded": int((~keep).sum())}

    # ---- per-class report and confusion matrices
    report = pd.DataFrame({
        "class": classes,
        "precision": m["per_class"]["precision"],
        "recall": m["per_class"]["recall"],
        "f1": m["per_class"]["f1"],
        "support": m["per_class"]["support"],
        "imagenet_counterpart": [FOOD101_TO_IMAGENET.get(c, "") for c in classes],
    })
    report.to_csv(out_dir / "classification_report.csv", index=False)
    in_imnet = report["imagenet_counterpart"] != ""
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    title = f"{meta['dataset']} / {cfg['experiment']['name']} / seed {seed}"
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix_normalized.png", normalize=True,
                          title=title + " (row-normalized)", annotate=False)

    # ---- predictions (top-5 kept for error analysis)
    conf = probs.max(axis=1)
    prob_true = probs[np.arange(len(y_true)), y_true]
    correct = y_true == y_pred
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    pd.DataFrame({
        "index": np.arange(len(y_true)), "entry": dataset.entries, "y_true": y_true, "y_pred": y_pred,
        "true_name": [classes[i] for i in y_true], "pred_name": [classes[i] for i in y_pred],
        "confidence": conf, "prob_true_class": prob_true, "correct": correct,
        "top5": [" ".join(map(str, r)) for r in top5_idx],
    }).to_csv(out_dir / "predictions.csv", index=False)

    # ---- error analysis
    confusions = top_confusions(cm, classes, top_k=25)
    confusions.to_csv(out_dir / "top_confusions.csv", index=False)
    errors = ~correct
    error_summary = {
        "n_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "errors_with_true_class_in_top5": int((errors & (top5_idx == y_true[:, None]).any(axis=1)).sum()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_errors": float(conf[errors].mean()) if errors.any() else None,
        "high_confidence_errors_conf_gt_0.9": int((errors & (conf > 0.9)).sum()),
        "top_confusions": confusions.head(10).to_dict("records"),
    }

    # ---- example grids (un-normalized centre crops)
    n_ex = int(ecfg.get("n_examples", 25))
    rng = np.random.default_rng(seed)
    correct_idx = np.where(correct)[0]
    wrong_idx = np.where(errors)[0]

    def grid(indices, filename, heading):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return
        titles = [f"T: {classes[y_true[i]]}\nP: {classes[y_pred[i]]} ({conf[i]:.2f})" for i in indices]
        colors = ["tab:green" if correct[i] else "tab:red" for i in indices]
        plot_image_grid([dataset.display_image(int(i)) for i in indices], titles, out_dir / filename,
                        ncols=5, suptitle=f"{title}: {heading}", title_colors=colors)

    if len(correct_idx):
        grid(np.sort(rng.choice(correct_idx, size=min(n_ex, len(correct_idx)), replace=False)),
             "examples_correct_random.png", "random correct predictions")
        grid(correct_idx[np.argsort(conf[correct_idx])[:n_ex]],
             "examples_correct_low_confidence.png", "difficult but correct (lowest confidence)")
    if len(wrong_idx):
        grid(np.sort(rng.choice(wrong_idx, size=min(n_ex, len(wrong_idx)), replace=False)),
             "examples_incorrect_random.png", "random incorrect predictions")
        grid(wrong_idx[np.argsort(-conf[wrong_idx])[:n_ex]],
             "examples_incorrect_high_confidence.png", "most confident errors")

    # ---- inference cost
    icfg = ecfg.get("inference", {})
    inference = measure_inference(
        model, dataset, device,
        batch_size=int(icfg.get("batch_size", 128)), warmup_iters=int(icfg.get("warmup_iters", 10)),
        timed_iters=int(icfg.get("timed_iters", 30)), latency_samples=int(icfg.get("latency_samples", 200)),
        amp=bool(icfg.get("amp", False)),
    )

    metrics = {
        "run_id": cfg.get("run", {}).get("run_id"),
        "dataset": meta["dataset"],
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"]["name"],
        "seed": seed,
        "split": split,
        "n_samples": int(len(y_true)),
        "loss": res["loss"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "top5_accuracy": top5,
        "ece": ece["ece"],
        "calibration_bins": ece["bins"],
        "bootstrap_ci": ci,
        "without_near_duplicates": without_dups,
        "macro_f1_imagenet_counterpart": float(report.loc[in_imnet, "f1"].mean()) if in_imnet.any() else None,
        "macro_f1_no_imagenet_counterpart": float(report.loc[~in_imnet, "f1"].mean()),
        "per_class": report.to_dict("records"),
        "error_summary": error_summary,
        "parameters": count_parameters(model),
        "gmacs_per_image": count_macs(model, meta["image_size"]) / 1e9,
        "inference": inference,
        "eval_amp": use_amp,
        "eval_pass_time_s": eval_pass_time,
        "checkpoint": {
            "file": str(ckpt_path.relative_to(run_dir)),
            "epoch": ckpt.get("epoch"),
            "selection_metric": ckpt.get("checkpoint_metric"),
            "val_metrics": ckpt.get("val_metrics"),
            "sha256": sha256_file(ckpt_path),
        },
        "split_file": meta["split_file"],
        "split_fingerprints": meta["split_fingerprints"],
        "git_at_eval": git_info(),
    }
    name = "test_metrics.json" if split == "test" else f"{split}_metrics.json"
    save_json(metrics, out_dir / name)
    log(f"[{split}] acc {m['accuracy']:.4f} (95% CI {ci['accuracy'][0]:.4f}-{ci['accuracy'][1]:.4f}) | "
        f"macroF1 {m['macro_f1']:.4f} (95% CI {ci['macro_f1'][0]:.4f}-{ci['macro_f1'][1]:.4f}) | "
        f"top5 {top5:.4f} | ECE {ece['ece']:.4f} | "
        f"throughput {inference['throughput_img_per_s']:.0f} img/s | "
        f"latency bs1 {inference['latency_ms_bs1_mean']:.2f} ms")
    return metrics
