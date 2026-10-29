"""Food-101: metadata, one-pass image index, fixed split, Dataset, transforms, DataLoaders.

Folder layout (official archive, and the pre-resized copy made by scripts/preresize.py):
  <root>/meta/classes.txt, train.txt, test.txt     entries look like "apple_pie/1005649"
  <root>/images/<class>/<id>.jpg

Split protocol (identical for every model):
  * official train list -> stratified train / val (StratifiedShuffleSplit, split_seed, val_fraction)
  * val images with a near-duplicate (pHash Hamming <= max_hamming) in train are moved to train
  * official test list  -> test, untouched; test images with a near-duplicate in train/val are
    listed in test_near_duplicates.txt (metrics are reported with and without them)
  * data/splits/{train,val,test}.txt + split.json are committed, so every run reads the same lists
  * the from-scratch baseline normalizes with mean/std of the TRAIN split only; pretrained models
    use the ImageNet statistics their weights were trained with
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.phash import hex_to_uint64, near_duplicate_pairs, phash
from src.utils.io import load_json, resolve_path, save_json
from src.utils.seed import seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- metadata
def read_meta(root):
    """Return (classes, official_train_entries, official_test_entries) from <root>/meta."""
    meta = resolve_path(root) / "meta"
    if not (meta / "classes.txt").exists():
        raise FileNotFoundError(f"{meta}/classes.txt not found. Run scripts/download_data.py "
                                "(and scripts/preresize.py) first, or fix data.image_root.")
    read = lambda name: [l.strip() for l in (meta / name).read_text().splitlines() if l.strip()]
    return read("classes.txt"), read("train.txt"), read("test.txt")


def image_path(root, entry: str) -> Path:
    return resolve_path(root) / "images" / f"{entry}.jpg"


def entry_label(entries, classes) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[e.split("/", 1)[0]] for e in entries], dtype=np.int64)


def fingerprint(entries) -> str:
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- image index
def index_one(args):
    """Per-image record used by the split, normalization and EDA (one decode per image)."""
    entry, image_root, raw_root = args
    rec = {"entry": entry}
    if raw_root is not None:
        p = image_path(raw_root, entry)
        with Image.open(p) as im:                       # header only, no pixel decode
            rec.update(raw_w=im.width, raw_h=im.height, raw_mode=im.mode, raw_bytes=p.stat().st_size)
    p = image_path(image_root, entry)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    luma = x @ np.array([0.299, 0.587, 0.114])
    rec.update(
        w=rgb.width, h=rgb.height, bytes=p.stat().st_size, n_px=x.shape[0] * x.shape[1],
        sum_r=x[..., 0].sum(), sum_g=x[..., 1].sum(), sum_b=x[..., 2].sum(),
        sumsq_r=(x[..., 0] ** 2).sum(), sumsq_g=(x[..., 1] ** 2).sum(), sumsq_b=(x[..., 2] ** 2).sum(),
        brightness=float(luma.mean()), contrast=float(luma.std()),
        saturation=float((x.max(axis=2) - x.min(axis=2)).mean()),
        phash=f"{phash(rgb):016x}",
    )
    return rec


def load_index(path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image index {path} not found. Run scripts/index_images.py")
    return pd.read_csv(path)


def channel_stats(index: pd.DataFrame):
    n = index["n_px"].sum()
    mean = np.array([index[f"sum_{c}"].sum() for c in "rgb"]) / n
    sq = np.array([index[f"sumsq_{c}"].sum() for c in "rgb"]) / n
    return mean.tolist(), np.sqrt(np.maximum(sq - mean ** 2, 1e-12)).tolist()


# --------------------------------------------------------------------------- split
def split_paths(split_dir):
    d = resolve_path(split_dir)
    return {"json": d / "split.json", **{s: d / f"{s}.txt" for s in ("train", "val", "test")},
            "test_near_duplicates": d / "test_near_duplicates.txt"}


def create_split(image_root, split_dir, index_path, split_seed: int = 42, val_fraction: float = 0.1,
                 max_hamming: int = 4):
    classes, official_train, official_test = read_meta(image_root)
    y = entry_label(official_train, classes)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))
    train = sorted(official_train[i] for i in tr_idx)
    val = sorted(official_train[i] for i in va_idx)
    test = list(official_test)

    index = load_index(index_path).set_index("entry")
    missing = set(official_train + official_test) - set(index.index)
    if missing:
        raise ValueError(f"{len(missing)} images missing from {index_path}; rebuild the index.")
    h = lambda entries: hex_to_uint64(index.loc[entries, "phash"])

    # val images that are near-duplicates of a train image -> train (prevents selection leakage)
    vi, _, _ = near_duplicate_pairs(h(val), h(train), max_hamming)
    moved = sorted({val[i] for i in vi})
    if moved:
        moved_set = set(moved)
        val = [e for e in val if e not in moved_set]
        train = sorted(train + moved)
    # test images with a near-duplicate in train or val -> flagged, kept
    trainval = train + val
    ti, tj, td = near_duplicate_pairs(h(test), h(trainval), max_hamming)
    flagged = sorted({test[i] for i in ti})
    pairs = pd.DataFrame({"test": [test[i] for i in ti], "match": [trainval[j] for j in tj],
                          "match_split": ["train" if j < len(train) else "val" for j in tj], "hamming": td})

    mean, std = channel_stats(index.loc[train].reset_index())
    counts = {name: np.bincount(entry_label(entries, classes), minlength=len(classes)).tolist()
              for name, entries in (("train", train), ("val", val), ("test", test))}
    payload = {
        "dataset": "food101",
        "classes": classes,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "stratified": True,
        "split_unit": "image",
        "protocol": "official train list -> stratified train/val; official test list -> test",
        "near_duplicates": {
            "method": "64-bit pHash, Hamming distance",
            "max_hamming": max_hamming,
            "val_moved_to_train": moved,
            "n_val_moved_to_train": len(moved),
            "n_test_flagged": len(flagged),
            "test_flagged_file": "test_near_duplicates.txt",
        },
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "counts": counts,
        "normalization": {"mean": mean, "std": std,
                          "computed_on": "train split only, pre-resized images (shorter side 256), pixels in [0,1]"},
        "fingerprints": {"train": fingerprint(train), "val": fingerprint(val), "test": fingerprint(test)},
    }
    paths = split_paths(split_dir)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    for name, entries in (("train", train), ("val", val), ("test", test), ("test_near_duplicates", flagged)):
        paths[name].write_text("\n".join(entries) + ("\n" if entries else ""))
    pairs.to_csv(paths["json"].parent / "test_near_duplicate_pairs.csv", index=False)
    save_json(payload, paths["json"])
    return payload, paths


def load_split(split_dir):
    paths = split_paths(split_dir)
    if not paths["json"].exists():
        raise FileNotFoundError(f"{paths['json']} not found. Run scripts/make_splits.py")
    split = load_json(paths["json"])
    read = lambda p: [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    lists = {s: read(paths[s]) for s in ("train", "val", "test", "test_near_duplicates")}
    for s in ("train", "val", "test"):
        if fingerprint(lists[s]) != split["fingerprints"][s]:
            raise ValueError(f"{paths[s]} does not match the fingerprint in split.json")
    return split, lists


def stratified_subset(entries, labels, fraction: float, seed: int):
    """Keep round(fraction * n_c) images of every class c (E3 data-efficiency sweep)."""
    if fraction >= 1.0:
        return list(entries), np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep.append(rng.permutation(idx)[:max(1, int(round(fraction * len(idx))))])
    keep = np.sort(np.concatenate(keep))
    return [entries[i] for i in keep], np.asarray(labels)[keep]


# --------------------------------------------------------------------------- dataset
def build_transforms(dcfg: dict, mean, std, train: bool):
    size = int(dcfg.get("image_size", 224))
    resize = int(dcfg.get("resize", 256))
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose([T.Resize(resize), T.CenterCrop(size), *norm])
    a = dcfg.get("augment") or {}
    ops = [T.RandomResizedCrop(size, scale=tuple(a.get("crop_scale", (0.35, 1.0))))]
    if a.get("hflip", True):
        ops.append(T.RandomHorizontalFlip())
    cj = float(a.get("color_jitter", 0.0) or 0.0)
    if cj > 0:
        ops.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    return T.Compose([*ops, *norm])


class Food101Dataset(Dataset):
    def __init__(self, entries, labels, image_root, transform, display_size: int = 224, resize: int = 256):
        self.entries = list(entries)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.image_root = resolve_path(image_root)
        self.transform = transform
        self.display = T.Compose([T.Resize(resize), T.CenterCrop(display_size)])

    def __len__(self) -> int:
        return len(self.entries)

    def load(self, idx) -> Image.Image:
        with Image.open(image_path(self.image_root, self.entries[idx])) as im:
            return im.convert("RGB")

    def __getitem__(self, idx):
        return self.transform(self.load(idx)), self.labels[idx]

    def display_image(self, idx) -> np.ndarray:
        """Un-normalized centre crop (H, W, 3) uint8, as seen by the model at eval time."""
        return np.asarray(self.display(self.load(idx)))


def build_datasets(cfg: dict, logger=None):
    dcfg = cfg["data"]
    split, lists = load_split(dcfg["split_dir"])
    classes = split["classes"]
    if dcfg.get("normalization", "imagenet") == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = split["normalization"]["mean"], split["normalization"]["std"]

    train_entries, train_labels = lists["train"], entry_label(lists["train"], classes)
    fraction = float(dcfg.get("train_fraction", 1.0))
    train_entries, train_labels = stratified_subset(train_entries, train_labels, fraction,
                                                    int(dcfg.get("subset_seed", 42)))
    if fraction < 1.0 and logger:
        logger.info("Using %.0f%% of the train split: %d images", 100 * fraction, len(train_entries))

    root = dcfg["image_root"]
    eval_tf = build_transforms(dcfg, mean, std, train=False)
    datasets = {
        "train": Food101Dataset(train_entries, train_labels, root, build_transforms(dcfg, mean, std, train=True)),
        "val": Food101Dataset(lists["val"], entry_label(lists["val"], classes), root, eval_tf),
        "test": Food101Dataset(lists["test"], entry_label(lists["test"], classes), root, eval_tf),
    }
    meta = {
        "dataset": "food101",
        "classes": classes,
        "num_classes": len(classes),
        "image_size": int(dcfg.get("image_size", 224)),
        "split_file": str(split_paths(dcfg["split_dir"])["json"]),
        "split_fingerprints": {**split["fingerprints"], "train_used": fingerprint(train_entries)},
        "train_fraction": fraction,
        "sizes": {k: len(v) for k, v in datasets.items()},
        "normalization": {"mean": list(mean), "std": list(std), "source": dcfg.get("normalization", "imagenet")},
        "test_near_duplicates": lists["test_near_duplicates"],
    }
    return datasets, meta


def build_dataloaders(cfg: dict, seed: int, device: torch.device, logger=None):
    datasets, meta = build_datasets(cfg, logger=logger)
    nw = int(cfg["data"].get("num_workers", 0))
    common = {
        "num_workers": nw,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if nw > 0 else None,
        "persistent_workers": nw > 0,
        "prefetch_factor": 4 if nw > 0 else None,
    }
    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(cfg["train"]["batch_size"])
    ebs = int(cfg["train"].get("eval_batch_size", bs))
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=bs, shuffle=True, drop_last=True,
                            generator=g, **common),
        "val": DataLoader(datasets["val"], batch_size=ebs, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=ebs, shuffle=False, **common),
    }
    return loaders, datasets, meta
