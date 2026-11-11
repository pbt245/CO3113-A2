# CO3133 Assignment 2 - Food-101 Image Classification

Deep Learning on Large-Scale Data and Specialized Tasks - Group G-roundhog, Semester 261, HCMUT.
Fine-grained 101-way food recognition: a from-scratch CNN baseline against ImageNet-pretrained
ResNet-50 / EfficientNet-B0 (ConvNeXt-Tiny optional), plus controlled fine-tuning experiments.

**Current milestone: M3 - Final** (all main models and controlled experiments over 3 seeds, error taxonomy, compute cost).

- Dataset proposal (M1, approved): [PROPOSAL.md](PROPOSAL.md)
- M3 final report: [docs/REPORT_M3.md](docs/REPORT_M3.md) (M2 draft: [docs/REPORT_M2.md](docs/REPORT_M2.md))
- Assignment page: <https://pbt245.github.io/CO3133-DLIA/assignments/assignment2.html>
- Report (PDF) / slides / video: `[TODO: links]`
- Checkpoints: `[TODO: link to checkpoints_food101.zip]` (section 7)

| Item | Choice |
| --- | --- |
| Track | Image classification |
| Dataset | Food-101 (101 classes, 101,000 images) |
| Split | official train → stratified 90 / 10 train / val (split seed 42), official test untouched; split unit = image; pHash near-duplicate checks |
| Metrics | macro-F1 (primary), top-1 accuracy, top-5, ECE, per-class F1, compute cost |
| Models | B0 SimpleCNN (scratch), P1 ResNet-50, P2 EfficientNet-B0, P3 ConvNeXt-T (optional) |
| Controlled experiments | E1 frozen backbone vs full fine-tune (mandatory), E2 label smoothing, E3 data fraction (optional) |

## Repository structure

```
configs/
  common.yaml                    shared defaults (split, augmentation, training protocol, evaluation)
  food101/_base.yaml             dataset entry point
  food101/{b0_simple_cnn,p1_resnet50,p2_efficientnet_b0,p3_convnext_tiny}.yaml
  food101/ablations/             E1, E2, E3 configs (reference, changed factor, hypothesis, decision criterion)
data/
  splits/                        train/val/test lists, split.json, test near-duplicate list  (committed)
  meta/image_index.csv.gz        per-image sizes, pixel sums, pHash                          (committed)
  noise_audit/label_audit.csv    manual label-noise audit sheet                              (committed)
  error_review/p1_error_review.csv  manual review of 125 P1 test errors                      (committed)
notebooks/
  kaggle_prepare.ipynb           CPU notebook: pre-resize (no GPU quota)
  kaggle_train.ipynb             GPU notebook: timing runs, training, comparison, result bundle
scripts/
  smoke_test.py                  offline unit + end-to-end check on synthetic data
  download_data.py               official archive (torchvision, md5-checked)
  preresize.py                   one-time resize to shorter side 256 px
  index_images.py                one decode per image -> data/meta/image_index.csv.gz
  make_splits.py                 stratified split + near-duplicate handling + train-only mean/std
  eda.py                         class balance, geometry, pixel stats, sample grids, near-duplicates
  noise_audit.py                 make / score the manual label-noise audit (Wilson 95 % CI)
  train.py                       train one config + seed (resumable), then evaluate on test
  evaluate.py                    re-evaluate a finished run
  run_experiments.py             run many configs x seeds (skips finished, resumes unfinished)
  compare.py                     mean ± std tables, decision rule, McNemar, compute cost, E3, figures
  error_analysis.py              error taxonomy, model error overlap, calibration, manual error review
  gradcam.py                     Grad-CAM panels for selected test images [OPTIONAL]
  pack_results.py                zip results / the checkpoint release (best.pt of every run)
src/
  data/food101.py                metadata, index, split, Dataset, transforms, DataLoaders
  data/phash.py                  64-bit pHash + Hamming search
  data/imagenet_overlap.py       Food-101 classes with an ImageNet-1k counterpart
  data/food_groups.py            13 culinary groups of the 101 classes (error taxonomy)
  models/                        simple_cnn.py (B0), pretrained.py (torchvision backbones + new head)
  engine/                        loops.py trainer.py evaluate.py metrics.py inference.py
  utils/ viz/                    viz/gradcam.py: Grad-CAM with hooks
```

Self-implemented: B0 architecture, data pipeline (split, near-duplicate handling, Dataset), pHash,
training / evaluation loops with resume, discriminative-LR parameter groups, frozen-backbone mode,
metrics aggregation (bootstrap CI, ECE, McNemar, decision rule). From libraries: torchvision backbones
and ImageNet weights, `torch.nn` layers, AdamW, torchvision transforms, scikit-learn (stratified split,
F1, confusion matrix), SciPy (DCT, McNemar / Wilson statistics).

## 1. Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
# GPU: install the CUDA build of torch/torchvision first (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt             # or: pip install -r requirements-lock.txt (exact versions of the reported runs)
python scripts/smoke_test.py                # ~2 min on CPU, no download needed
```

## 2. Dataset preparation

The split is generated once (steps 3-4) and then committed (`data/splits/`, `data/meta/image_index.csv.gz`);
every later machine only needs steps 1-2 and reuses the committed lists, whose fingerprints are
checked at load time. Rebuilding steps 3-4 is only needed to reproduce the split itself.

```bash
python scripts/download_data.py                                   # 1. ~5 GB -> data/raw/food-101
python scripts/preresize.py --workers 8                           # 2. -> data/resized/food-101-256 (~2 GB)
python scripts/index_images.py --workers 8                        # 3. -> data/meta/image_index.csv.gz
python scripts/make_splits.py                                     # 4. -> data/splits/
python scripts/eda.py                                             # 5. -> outputs/eda/food101/
python scripts/noise_audit.py make                                # 6. sheets + CSV; fill it in, then:
python scripts/noise_audit.py score
```

Split protocol: official train list → `StratifiedShuffleSplit(test_size=0.1, random_state=42)`.
Validation images whose pHash is within Hamming distance 4 of a train image are moved to train.
Official test list → test, untouched; test images with a near-duplicate in train/val are listed in
`data/splits/test_near_duplicates.txt` and every evaluation also reports metrics without them.
B0 normalizes with mean/std of the train split only; pretrained models use ImageNet statistics.
Every run records the split fingerprints.

## 3. Training and evaluation

Same commands on any machine (RTX 3060, Kaggle T4, ...). On Kaggle use
[notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb), which wraps them.

```bash
# 1-epoch timing run (300 steps): measured img/s for the compute estimate
python scripts/train.py --config configs/food101/p1_resnet50.yaml --no-eval \
    --set train.epochs=1 train.max_steps_per_epoch=300 experiment.name=timing_p1 experiment.group=debug

# M3 - every reported run (finished runs are skipped, interrupted ones resumed); ~13 GPU-h on an RTX 3060
S="--set data.num_workers=10"
python scripts/run_experiments.py --models p2_efficientnet_b0 e1_resnet50_linear_probe --seeds 42 123 2026 $S
python scripts/run_experiments.py --models b0_simple_cnn p1_resnet50 --seeds 42 123 2026 $S
python scripts/run_experiments.py --models e2_resnet50_label_smoothing --seeds 42 123 2026 $S          # optional
python scripts/run_experiments.py --models e3_b0_simple_cnn_frac10 e3_b0_simple_cnn_frac25 e3_b0_simple_cnn_frac50 \
    e3_p1_resnet50_frac10 e3_p1_resnet50_frac25 e3_p1_resnet50_frac50 p3_convnext_tiny --seeds 42 $S  # optional

# images somewhere else (e.g. Kaggle input) -> override per run
python scripts/train.py --config configs/food101/p1_resnet50.yaml --set data.image_root=/kaggle/input/x/food-101-256

# resume an interrupted run (Kaggle 12 h limit or train.max_hours)
python scripts/train.py --config configs/food101/p1_resnet50.yaml --seed 42 --resume

# re-evaluate a run
python scripts/evaluate.py --run-dir outputs/runs/food101/p1_resnet50/seed42
```

Fine-tuning protocol (P1-P3): `RandomResizedCrop(224, scale=(0.35, 1))` + horizontal flip + colour
jitter 0.1; eval `Resize(256)` + `CenterCrop(224)`; AdamW, weight decay 0.05 (not on biases / norm
parameters); discriminative LR - new head 1e-3, backbone 1e-4; 1 warm-up epoch then cosine decay to 1 %;
10 epochs; batch 64; fp16 mixed precision; gradient clipping 1.0. B0: same pipeline, one LR 1e-3,
30 epochs, train-split normalization. E1: backbone frozen (BatchNorm in eval mode), only the new fc
is trained, everything else identical to P1.
**Checkpoint selection rule: highest validation macro-F1** (`checkpoints/best.pt`); early stopping
after 4 epochs without improvement. The test set is used once per selected checkpoint.

## 4. Analysis

```bash
python scripts/compare.py                        # outputs/comparison/food101/: summary, decisions, compute_cost,
                                                 #   data_efficiency, mcnemar, curves, per-class F1
python scripts/error_analysis.py taxonomy        # outputs/error_analysis/food101/: taxonomy.md, calibration.png, ...
python scripts/error_analysis.py review-make     # sheets + CSV for the manual error review (already filled in)
python scripts/error_analysis.py review-score    # -> review_result.md
python scripts/gradcam.py --models b0_simple_cnn p1_resnet50 p2_efficientnet_b0 p3_convnext_tiny   # [OPTIONAL]
python scripts/pack_results.py                   # zip of eda/comparison/runs (no checkpoints)
python scripts/pack_results.py --checkpoints-only  # outputs/checkpoints_food101.zip + CHECKPOINTS_SHA256.txt
```

Error taxonomy: every test error goes to exactly one bucket - *consensus* (all pretrained models predict the
same wrong class), *sibling* (true and predicted class in the same culinary group, `src/data/food_groups.py`),
*near-miss* (true class in the top 5) or *gross*. The manual review classifies 75 random and the 50 most
confident P1 errors as `label_wrong` / `ambiguous` / `model_error` (Wilson 95 % CI).

Decision rule (proposal section 4): A is better than B only if the gap in mean test macro-F1 over
3 seeds exceeds the larger seed std **and** McNemar on seed 42 gives p < 0.05. `decisions.md` marks
comparisons with fewer than 2 seeds per arm as preliminary.

## 5. Traceability

Each run folder `outputs/runs/food101/<experiment>/seed<seed>/` contains the resolved `config.yaml`
(with `run_id`, CLI overrides and resume times), `env.json` (hardware, CUDA/cuDNN, library versions,
git commit), `train.log`, `history.csv`, `train_summary.json`, `model_architecture.txt`
(parameter count, GMACs) and `eval/test_metrics.json` (split fingerprints, checkpoint epoch and SHA-256).
`outputs/comparison/food101/runs.csv` maps every reported number to its run_id, commit, split
fingerprint and checkpoint hash.

Seeds: split seed 42 (fixed); training seeds 42, 123, 2026; E3 subset seed 42.
`deterministic: true` in the config enables deterministic kernels at some speed cost; otherwise GPU
results vary slightly between runs.

## 6. Hardware

| Machine | Used for | Details |
| --- | --- | --- |
| MacBook Air M2, 24 GB | data preparation, EDA, smoke test | macOS, CPU only for the reported numbers |
| Desktop | **every reported run** (training, evaluation, timing) | NVIDIA GeForce RTX 3060 12 GB, 12-core CPU, 31 GB RAM, Arch Linux, Python 3.12.11, torch 2.14.1+cu126, torchvision 0.29.1, CUDA 12.6, cuDNN 9.10 |
| Kaggle notebook | supported, not used for reported numbers | NVIDIA Tesla T4 16 GB, 4 vCPU ([notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb)) |

The GPU of every reported run is listed in `runs.csv` (`gpu` column).

## 7. Checkpoints

Checkpoints are not stored in git. Download `checkpoints_food101.zip`: `[TODO: link]` and unzip it **at the
repository root**: it holds `outputs/runs/food101/<experiment>/seed<seed>/checkpoints/best.pt` for every reported
run, next to the committed run files, plus `CHECKPOINTS_SHA256.txt`. Then:

```bash
sha256sum -c CHECKPOINTS_SHA256.txt                                   # hashes also in each eval/test_metrics.json
python scripts/evaluate.py --run-dir outputs/runs/food101/p1_resnet50/seed42
```

`best.pt` = weights of the epoch with the highest validation macro-F1 (no optimizer state). To rebuild instead,
rerun the training commands above at the commit listed in `outputs/comparison/food101/runs.csv`.

## Members

| Name | Student ID | GitHub | Email |
| --- | --- | --- | --- |
| Phan Bá Thanh | 2353084 | [pbt245](https://github.com/pbt245) | thanh.phancsbk@hcmut.edu.vn |
| Trần Công Hoàng Phước | 2352966 | [BenjaminPhuoc](https://github.com/BenjaminPhuoc) | phuoc.tranbenjamin@hcmut.edu.vn |
