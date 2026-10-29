# Assignment 2 - M2 Draft Report: Food-101 Image Classification

Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
Group G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
Track: image classification. Proposal (M1): [PROPOSAL.md](../PROPOSAL.md), approved.

**Status of this draft.** The EDA, the split, the simple baseline (B0) and the first pretrained model (P1) are done. All numbers come from **one training seed (42)**, so every comparison is *preliminary* under the decision rule of the proposal (section 4 needs at least 2 seeds per arm, and 3 in the final report). The mandatory controlled experiment (E1), the second pretrained model (P2), extra seeds, the error taxonomy, the full compute-cost analysis, slides, video and the Pages update belong to M3. The manual label-noise audit is done (section 1.4): about 6.7 % of train labels are wrong.

---

## 1. Problem and data description

### 1.1 Problem

Single-label, 101-way classification of dish photographs (Food-101, Bossard et al., ECCV 2014). Input: one RGB image. Output: logits over 101 classes, trained with cross-entropy. The task is fine-grained: the same dish varies strongly in plating, lighting and framing, while different dishes can look alike (`steak` / `filet_mignon` / `prime_rib` / `pork_chop`).

### 1.2 Dataset

| Quantity | Value |
| --- | --- |
| Classes | 101 |
| Images | 101,000 (750 official train + 250 official test per class) |
| Raw image size | max side 512 px for 100 % of images; width mean 496, height mean 475; aspect w/h median 1.00 (p5 0.75, p95 1.50) |
| Orientation | landscape 25.4 %, square 62.9 %, portrait 11.7 % |
| Colour modes | RGB 100,997; grayscale 3 |
| Images with shorter side < 224 px | 0.01 % |

Spec thresholds (classification: at least 5 classes, at least 5,000 training samples): met (101 classes, 68,189 training images). Source and licence: official page, images are not redistributed by us.

Known biases (from the proposal, confirmed or still open):
- **Train labels are noisy by design** (authors cleaned only the test set). A manual audit of 200 train images measured a wrong-label rate of **6.7 %** (95 % CI 4.0-11.2 %), section 1.4.
- Photos come from a photo-sharing site, mostly Western restaurant dishes; results do not transfer to "food in general".
- Only **7 of 101** classes have an ImageNet-1k counterpart (`src/data/imagenet_overlap.py`), so the pretrained model's head start from ImageNet food knowledge is limited to a few classes.

### 1.3 EDA (full tables: [outputs/eda/food101/eda_summary.md](../outputs/eda/food101/eda_summary.md))

| Split | Images | Min / class | Max / class | Max/min |
| --- | ---: | ---: | ---: | ---: |
| train | 68,189 | 675 | 676 | 1.001 |
| val | 7,561 | 74 | 75 | 1.014 |
| test | 25,250 | 250 | 250 | 1.000 |

- **No class imbalance.** Accuracy and macro-F1 are therefore expected to be close, and they are (section 3).
- **Pixel statistics** are almost identical across splits (mean RGB 0.540 / 0.442 / 0.344 on train, 0.543 / 0.445 / 0.346 on test; brightness 0.463 vs 0.466; saturation 0.231 vs 0.230), so there is no visible covariate shift between train and test. Train-only mean/std (used by B0): mean [0.5402, 0.4420, 0.3441], std [0.2704, 0.2724, 0.2773].
- Darkest classes: bibimbap, miso_soup, mussels. Brightest: spaghetti_carbonara, risotto.
- Figures: [class_distribution.png](../outputs/eda/food101/class_distribution.png), [image_sizes.png](../outputs/eda/food101/image_sizes.png), [pixel_stats.png](../outputs/eda/food101/pixel_stats.png), [samples_all_classes.png](../outputs/eda/food101/samples_all_classes.png), [confusable_pairs.png](../outputs/eda/food101/confusable_pairs.png).

### 1.4 Label-noise audit

Protocol (proposal 2.4): 20 random classes x 10 random train images (seed 42), judged by a group member as `ok` / `wrong` / `unsure`; wrong-label rate with a Wilson 95 % CI. Verdicts: [data/noise_audit/label_audit.csv](../data/noise_audit/label_audit.csv), sheets in `outputs/eda/food101/noise_audit/`, scoring in [audit_result.md](../outputs/eda/food101/noise_audit/audit_result.md).

| Judged | Unsure (excluded) | Wrong | Wrong-label rate | Wilson 95 % CI |
| ---: | ---: | ---: | ---: | --- |
| 193 | 7 | 13 | **6.7 %** | 4.0-11.2 % |

- Wrong labels are concentrated in a few classes: `filet_mignon` 4 of 10, `miso_soup` 3, `chicken_curry` 2, and one each in `frozen_yogurt`, `macaroni_and_cheese`, `pancakes`, `red_velvet_cake`. Thirteen of the 20 audited classes had no wrong label.
- The sample is 200 images from 20 classes, so the per-class pattern is anecdotal and the overall rate is an estimate with a wide interval. The `note` column of the CSV was left empty, so the reasons for individual verdicts are not recorded, and one rater judged every image (no inter-rater agreement).
- The proposal's claim that the test set is cleaner than the train set was **not** checked: only train images were audited.

### 1.5 Splitting and leakage prevention

| Split | Source | Images |
| --- | --- | ---: |
| Train | official train, minus val, plus 14 moved val images | 68,189 |
| Validation | stratified 10 % of official train, minus 14 moved | 7,561 |
| Test | official test, untouched | 25,250 |

- Split unit: **image** (Food-101 has no uploader / restaurant id). Stratified by class, `StratifiedShuffleSplit`, split seed **42**, identical lists for every model. Fingerprints: train `8f130b6f7c76d577`, val `bdded6761644591e`, test `101fc179791da757` (checked at load time, stored in every run).
- Near-duplicates (64-bit pHash, Hamming <= 4), over the whole dataset: test-train 39 pairs, test-val 9, train-train 88, val-val 1, test-test 6; 4 pairs carry different labels. **14 val images** that duplicated a train image were moved to train (so val is 7,561, not 7,575 as in the proposal). **47 test images (0.19 %)** are near-duplicates of train/val: kept for comparability with published numbers, and every evaluation also reports the metrics without them (B0 71.26 -> 71.25, P1 86.38 -> 86.37 macro-F1, so no effect).
- Remaining leakage risk: same-uploader photos that are not pHash-similar cannot be detected (limitation).
- Preprocessing: images pre-resized once to shorter side 256 px (JPEG q95, `data/resized/`). Train: `RandomResizedCrop(224, scale 0.35-1)`, horizontal flip, colour jitter 0.1. Val/test: `Resize(256)` + `CenterCrop(224)`. Pretrained models use ImageNet mean/std; B0 uses the train-only statistics above.
- Batch examples: [samples_all_classes.png](../outputs/eda/food101/samples_all_classes.png) and the `examples_*.png` files in each run's `eval/` folder.

---

## 2. Methodology

### 2.1 Pipeline

```
raw JPEG (<=512 px) -> pre-resize (short side 256) -> index + pHash -> stratified split (+dup check)
   -> Dataset / DataLoader (augment: RRC 224, flip, jitter; fp16 AMP)
   -> model (B0 | P1) -> cross-entropy loss -> AdamW + warm-up/cosine, grad clip 1.0
   -> per-epoch validation (macro-F1) -> best-val checkpoint -> test (once) + compute cost
```

### 2.2 Models

| ID | Model | Init | Params | GMACs @224 | Role |
| --- | --- | --- | ---: | ---: | --- |
| B0 | SimpleCNN: 4 blocks of (2 x conv3x3-BN-ReLU) + max-pool, channels 32-64-128-256, global avg-pool, dropout 0.3, linear head | random | 1,199,173 | 2.59 | simple baseline (mandatory) |
| P1 | ResNet-50 (torchvision `IMAGENET1K_V2`), new 101-way `fc` | ImageNet-1k | 23,714,981 | 4.09 | pretrained model (mandatory) |

The proposal estimated B0 at about 1 GFLOP; the measured value is 2.59 GMACs because the first block runs two convolutions at full 224 x 224 resolution. B0 is smaller in parameters (20x) but not in compute (0.63x of P1).

### 2.3 Training setup (identical for both models unless noted)

| Item | Value |
| --- | --- |
| Optimizer | AdamW, weight decay 0.05 (not on biases / norm parameters) |
| Learning rate | P1: head 1e-3, backbone 1e-4 (discriminative). B0: 1e-3 for all layers |
| Schedule | 1 warm-up epoch, cosine decay to 1 % |
| Epochs | P1 10, B0 30 (no early stop triggered in either run) |
| Batch size | 64, fp16 mixed precision, grad clip 1.0 |
| Checkpoint rule | best validation macro-F1 (P1 epoch 10, B0 epoch 30); test evaluated once per checkpoint |
| Seed | 42 (training and split) |
| Hardware | NVIDIA GeForce RTX 3060 12 GB, 12-core CPU, Arch Linux; Python 3.12.11, torch 2.14.1+cu126, torchvision 0.29.1, CUDA 12.6, cuDNN 9.10 (full list in each `env.json`) |

Fine-tuning procedure for P1: all layers trained from the ImageNet weights, new head at a 10x higher learning rate than the backbone, same augmentation as B0.

### 2.4 Hypotheses for M3 (from the proposal; not tested yet)

E1 (mandatory): full fine-tuning beats a frozen backbone by at least 5 pp macro-F1. E2 (optional): label smoothing 0.1 improves macro-F1 by at least 0.5 pp and lowers ECE. E3 (optional): P1 keeps at least 85 % of its full-data F1 with 10 % of the data, B0 does not.

---

## 3. Implementation results (preliminary, seed 42)

### 3.1 Quantitative results on the test set

| | B0 SimpleCNN | P1 ResNet-50 |
| --- | ---: | ---: |
| Top-1 accuracy | 71.35 % | 86.36 % (95 % CI 85.92-86.80) |
| **Macro-F1** | **71.26 %** | **86.38 %** (95 % CI 85.93-86.78) |
| Top-5 accuracy | 91.40 % | 97.16 % |
| ECE (15 bins) | 0.081 | 0.047 |
| Macro-F1 without 47 flagged near-duplicates | 71.25 % | 86.37 % |
| Best validation macro-F1 | 66.52 % | 82.37 % |
| Parameters | 1.20 M | 23.71 M |
| Train time (total) | 77.1 min (30 epochs, 2.6 min/epoch, 442 img/s) | 37.8 min (10 epochs, 3.8 min/epoch, 301 img/s) |
| Inference throughput (batch 128, fp16) | 1769 img/s | 909 img/s |
| Latency (batch 1, mean) | 0.77 ms | 2.71 ms |

(Source: [outputs/comparison/food101/summary.md](../outputs/comparison/food101/summary.md); every number maps to a run id, commit and checkpoint hash in `runs.csv`.)

**Comparison.** P1 is 15.1 pp macro-F1 above B0; McNemar on the same 25,250 test images gives p < 0.001 (P1 right / B0 wrong on 4,643 images, the reverse on 852). The gap is large, but with a single seed the proposal's decision rule marks this as **preliminary**: there is no seed standard deviation to compare against yet. Macro-F1 and accuracy differ by under 0.1 pp, as expected for balanced classes.

**Cost.** P1 needs half the training time of B0 for 15 pp more macro-F1, because it converges in 10 epochs instead of 30, even though each epoch is 1.5x slower. B0 is 2x faster at inference and has 3.5x lower batch-1 latency. Measured training throughput differs from the proposal's assumptions (P1 301 vs 250 img/s, B0 442 vs 700 img/s) and the GPU is an RTX 3060, not a T4, so these timings are not comparable with Kaggle runs.

### 3.2 Training behaviour ([main_curves.png](../outputs/comparison/food101/main_curves.png), per-run `curves.png`)

- **P1** learns fast (val macro-F1 69.8 % after epoch 1, 80.2 % at epoch 5, 82.4 % at epoch 10). Validation loss stops improving after about epoch 5 (0.78 -> 0.77) while train loss keeps falling (0.47 -> 0.15) and train accuracy reaches 95.8 % against 82.3 % validation accuracy: a clear generalisation gap, i.e. P1 starts to overfit the training set. Train accuracy is measured on augmented crops, so the true gap is somewhat larger.
- **B0** is still improving at the end (val macro-F1 45.7 % at epoch 10, 66.5 % at epoch 30, with the cosine schedule flattening the last epochs). Its train accuracy (62.9 %) is below its validation accuracy (66.5 %) because of augmentation and dropout in training only. B0 is under-trained / capacity-limited rather than overfitting; 30 epochs may not be enough to reach its limit.
- Test scores are higher than validation scores for both models (P1 86.4 vs 82.4, B0 71.3 vs 66.5). Likely cause: the validation set is carved from the noisy train list (about 6.7 % wrong labels, section 1.4), while the test set was manually cleaned by the dataset authors. The direction and rough size fit, but we did not audit the validation or test labels, so this is not proven.

### 3.3 Qualitative results

Figures per run, in `outputs/runs/food101/<experiment>/seed42/eval/`: `examples_correct_random.png`, `examples_correct_low_confidence.png`, `examples_incorrect_random.png`, `examples_incorrect_high_confidence.png`, `confusion_matrix_normalized.png`.

### 3.4 Early error analysis (taxonomy and full analysis come in M3)

P1 makes 3,443 errors (13.6 %); the true class is in its top 5 for 2,727 of them (79 %). Mean confidence is 0.95 on correct and 0.67 on wrong predictions; 715 errors have confidence above 0.9.

- **Visually close dishes** dominate the confusions. P1: filet_mignon -> steak (41, 16.4 % of that class), chocolate_mousse -> chocolate_cake (33), steak -> prime_rib (29), beef_tartare -> tuna_tartare (20), dumplings -> gyoza (17). B0 has the same pairs (steak -> filet_mignon 53). Weakest classes: P1 steak 0.557, pork_chop 0.665, chocolate_mousse 0.692, apple_pie 0.710; B0 pork_chop 0.369, steak 0.384, apple_pie 0.413.
- **Confident errors that may be label problems.** In the 25 most confident P1 errors ([examples_incorrect_high_confidence.png](../outputs/runs/food101/p1_resnet50/seed42/eval/examples_incorrect_high_confidence.png)) several images look to us like the predicted class rather than the label (e.g. a tomato-and-mozzarella dish labelled `pizza`, predicted `caprese_salad`; a nigiri plate labelled `sashimi`, predicted `sushi`), and some show several dishes or a menu/display case. This is a visual impression from one grid; the measured train noise rate is 6.7 % (section 1.4), but test images were not audited; the test set is described as manually reviewed, so these cases may equally be genuinely ambiguous photos. M3 can audit these cases.
- **ImageNet overlap** ([imagenet_overlap.md](../outputs/comparison/food101/imagenet_overlap.md)): P1 macro-F1 is 88.4 % on the 7 classes with an ImageNet-1k counterpart and 86.2 % on the other 94 (+2.2 pp); B0 shows no such gap (-1.7 pp). With 7 classes the first number is noisy.

### 3.5 Limitations of this draft

- One seed: no variance estimate, comparisons are preliminary.
- B0 may be under-trained at 30 epochs; this is unresolved.
- P1's early-epoch LR/overfitting behaviour was not tuned (single protocol from the proposal).
- The label-noise audit covers 200 train images judged by one rater, and neither val nor test labels were audited (section 1.4).
- Runs were made with uncommitted changes in the working tree (the `env.json` of both runs has `dirty: true`, base commit `f4968df`); the code was not modified between runs, but the final M3 runs should be made from a tagged clean commit.
- Timings are for an RTX 3060 and are not comparable with the T4 estimates of the proposal.

---

## 4. Deviations from the proposal

| Item | Proposal | M2 |
| --- | --- | --- |
| Val / train size | 7,575 / 68,175 | 7,561 / 68,189 (14 near-duplicate val images moved to train) |
| B0 compute | about 1 GFLOP | 2.59 GMACs measured |
| Hardware | Kaggle T4 | local RTX 3060 12 GB (the README lists it as an alternative) |
| Throughput assumptions | P1 250, B0 700 img/s | measured P1 301, B0 442 img/s (RTX 3060, `num_workers=10`) |
| Seeds | 3 | 1 for M2 (3 planned for M3) |
| Label-noise audit | planned for M2 EDA | done: 13 / 193 wrong (6.7 %, CI 4.0-11.2 %), train only, one rater |

## 5. Reproduction (M2)

```bash
python scripts/download_data.py && python scripts/preresize.py && python scripts/index_images.py
python scripts/make_splits.py && python scripts/eda.py
python scripts/run_experiments.py --models b0_simple_cnn p1_resnet50 --seeds 42 --set data.num_workers=10
python scripts/compare.py
```

Checkpoints (`best.pt`) are in `outputs/runs/food101/<experiment>/seed42/checkpoints/` and are not stored in git; the public link is added at M3.
