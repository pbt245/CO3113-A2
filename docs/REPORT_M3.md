# Assignment 2 - Final Report (M1-M3): Food-101 Image Classification

Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
Group G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
Track: image classification. Proposal (M1): [PROPOSAL.md](../PROPOSAL.md), approved without conditions. M2 draft: [REPORT_M2.md](REPORT_M2.md).

**Summary.** On Food-101 (101 classes, 68,189 training images), a from-scratch CNN reaches 71.1 % test macro-F1. ImageNet-pretrained models reach 84.8 % (EfficientNet-B0), 86.0 % (ResNet-50) and 88.5 % (ConvNeXt-Tiny, optional, one seed). All but the ConvNeXt-Tiny figure are means over 3 seeds, with seed standard deviations of 0.1-0.2 pp.
- **E1 (mandatory controlled experiment).** Full fine-tuning beats a frozen backbone by 15.5 pp, so the hypothesis (at least 5 pp) is confirmed.
- **E2 (optional).** Label smoothing gains 0.39 pp. That passes the decision rule but misses the 0.5 pp target, and it doubles the calibration error instead of lowering it.
- **E3 (optional).** ResNet-50 is much more data-efficient than the baseline, but it keeps 79 % (not the predicted at least 85 %) of its F1 with 10 % of the data.
- **Error analysis.** Most of the most confident errors are not model failures. In a manual review of 50 of them, 36 % had a wrong test label and 46 % showed an ambiguous photo.

All 22 reported runs (12 configurations) were trained from one commit (`a12ccd6`), on one GPU type, with one data split.

Work type: `[MANDATORY]` items are B0, P1, P2, E1, the EDA, the error analysis and the compute cost. Everything marked `[OPTIONAL]` (P3, E2, E3, Grad-CAM) is an extension and does not replace a mandatory item.

### Assignment overview (milestones M1-M3)

This is the final report for the whole of Assignment 2.

| Milestone | Required | What we delivered |
| --- | --- | --- |
| M1 Proposal | dataset proposal (task, data, split, metrics, baseline, compute estimate); approval before implementation | [PROPOSAL.md](../PROPOSAL.md), submitted 2026-10-07: Food-101, stratified split with seed 42, macro-F1 + pre-registered decision rule, models B0/P1/P2 (+P3), experiments E1 (+E2, E3), compute estimate of about 14 T4 GPU-h. **Approved, no conditions.** |
| M2 Draft | simple baseline and pretrained model in progress; EDA complete; preliminary metrics | Complete EDA, near-duplicate check and label-noise audit (6.7 % wrong train labels). B0 and P1 with seed 42 on a local RTX 3060: preliminary macro-F1 71.3 % / 86.4 % ([REPORT_M2.md](REPORT_M2.md)). |
| M3 Final | controlled experiment, error analysis, compute cost, report, pages, video, checkpoints | All 12 configurations (22 runs) from one commit on one GPU type, the main ones with 3 seeds. Error taxonomy, manual error review, Grad-CAM, compute cost. This report, slides, video and the checkpoints. |

The M2 seed-42 runs were re-run in M3 so that every number and timing comes from the same hardware and code. The M2 numbers are within 0.35 pp of the M3 seed-42 reruns (B0 71.26 vs 71.31 %, P1 86.38 vs 86.05 %). All results below are from M3.

---

## 1. Problem and data description

### 1.1 Problem

Single-label, 101-way classification of dish photographs. Input: one RGB image, cropped to 224 x 224 for every model. Output: 101 logits, trained with cross-entropy.

The task is fine-grained:
- **High intra-class variance.** The same dish varies in plating, lighting and framing.
- **Low inter-class variance.** Different dishes look alike, e.g. `steak` / `filet_mignon` / `prime_rib` / `pork_chop`, or `chocolate_cake` / `chocolate_mousse`.

### 1.2 Dataset

**Food-101.** Bossard, Guillaumin & Van Gool, ECCV 2014. Photos from foodspotting.com, research use; we do not redistribute images.

| Quantity | Value |
| --- | --- |
| Classes | 101 (spec threshold: at least 5) |
| Images | 101,000: 750 official train + 250 official test per class |
| Our training images | 68,189 (spec threshold: at least 5,000; MNIST / Fashion-MNIST excluded: yes) |
| Raw image size | max side 512 px for 100 %; mean width 496, mean height 475; aspect ratio w/h median 1.00 (p5 0.75, p95 1.50) |
| Orientation | landscape 25.4 %, square 62.9 %, portrait 11.7 % |
| Colour modes | RGB 100,997, grayscale 3 |

**Biases and known properties**
- **Noisy training labels by design.** The authors cleaned only the test set. Our manual audit of 200 train images (20 random classes x 10 images) found **6.7 %** wrong labels (13 of 193 judged; Wilson 95 % CI 4.0-11.2 %). The wrong labels concentrate in `filet_mignon` (4/10), `miso_soup` (3/10) and `chicken_curry` (2/10).
- **Cultural bias.** Mostly Western restaurant dishes from one photo-sharing site, so results do not transfer to "food in general".
- **ImageNet overlap.** Only 7 of 101 classes have a same-dish ImageNet-1k counterpart (`src/data/imagenet_overlap.py`), e.g. pizza, hot_dog, ice_cream, guacamole.

### 1.3 EDA

Full tables: [outputs/eda/food101/eda_summary.md](../outputs/eda/food101/eda_summary.md).

| Split | Images | Min / class | Max / class | Max/min |
| --- | ---: | ---: | ---: | ---: |
| train | 68,189 | 675 | 676 | 1.001 |
| val | 7,561 | 74 | 75 | 1.014 |
| test | 25,250 | 250 | 250 | 1.000 |

- **No class imbalance.** As expected, accuracy and macro-F1 stay within 0.25 pp of each other for every run (section 3.2). We still report both, plus per-class F1.
- **Pixel statistics match across splits.** Mean RGB is 0.540 / 0.442 / 0.344 on train and 0.543 / 0.445 / 0.346 on test; brightness 0.463 vs 0.466; saturation 0.231 vs 0.230. So there is no visible covariate shift.
- **Normalisation.** B0 uses mean/std from the train split only: mean [0.5402, 0.4420, 0.3441], std [0.2704, 0.2724, 0.2773]. Pretrained models use the ImageNet statistics.
- **Figures:**
  - [class_distribution.png](../outputs/eda/food101/class_distribution.png)
  - [image_sizes.png](../outputs/eda/food101/image_sizes.png)
  - [pixel_stats.png](../outputs/eda/food101/pixel_stats.png)
  - [samples_all_classes.png](../outputs/eda/food101/samples_all_classes.png)
  - [confusable_pairs.png](../outputs/eda/food101/confusable_pairs.png)
  - [near_duplicates_examples.png](../outputs/eda/food101/near_duplicates_examples.png)

### 1.4 Splitting and leakage prevention

| Split | Source | Images |
| --- | --- | ---: |
| Train | official train minus val, plus 14 moved val images | 68,189 |
| Validation | stratified 10 % of official train, minus 14 moved | 7,561 |
| Test | official test, untouched | 25,250 |

- **Split unit and protocol.**
  - Split unit: the **image** (Food-101 has no uploader or restaurant id).
  - Split: `StratifiedShuffleSplit(test_size=0.1, random_state=42)`, stratified by class.
  - The lists are committed in `data/splits/`, so every model uses byte-identical splits. Fingerprints are checked at load time and stored in every run: train `8f130b6f7c76d577`, val `bdded6761644591e`, test `101fc179791da757`.
- **Near-duplicates.** We computed a 64-bit perceptual hash (pHash) of every image and flagged pairs with Hamming distance at most 4.
  - 14 validation images that duplicated a train image were moved to train.
  - 47 test images (0.19 %) duplicate a train or val image. They are kept, for comparability with published numbers. Every evaluation also reports macro-F1 without them, and this changes no result by more than 0.02 pp (table 3.2).
- **Test-set discipline.** The test set is evaluated once per selected checkpoint and is never used for any decision. Hyper-parameters were not tuned at all: the protocol from the proposal was run unchanged.
- **Preprocessing.** Images are pre-resized once to shorter side 256 px (JPEG q95).
  - Train: `RandomResizedCrop(224, scale 0.35-1)`, horizontal flip, colour jitter 0.1.
  - Val/test: `Resize(256)` + `CenterCrop(224)`.
- **Batch examples.** See the `examples_*.png` files in every run's `eval/` folder (the eval-time centre crops with predictions).

---

## 2. Methodology

### 2.1 Pipeline

```
Food-101 JPEG (<= 512 px)
  -> pre-resize (short side 256) -> image index + pHash -> stratified split + near-duplicate handling   (once, CPU)
  -> Dataset / DataLoader (train: RandomResizedCrop 224, flip, colour jitter; eval: Resize 256 + CenterCrop 224)
  -> model (B0 | P1 | P2 | P3; E1 = frozen P1 backbone)
  -> cross-entropy (E2: label smoothing 0.1)
  -> AdamW, discriminative LR, 1 warm-up epoch + cosine, fp16 AMP, grad clip 1.0
  -> per-epoch validation (macro-F1) -> best-val checkpoint
  -> test once: acc / macro-F1 / top-5 / ECE / bootstrap CI / per-class / confusions / inference cost
  -> compare.py (mean ± std over seeds, decision rule, McNemar) -> error_analysis.py, gradcam.py
```

### 2.2 Models

| ID | Model | Init | Params (trainable) | GMACs @224 | Role |
| --- | --- | --- | ---: | ---: | --- |
| B0 | SimpleCNN, 4 blocks of (2 x conv3x3-BN-ReLU) + max-pool (channels 32-64-128-256), GAP, dropout 0.3, linear head | random | 1.20 M | 2.59 | simple baseline `[MANDATORY]` |
| P1 | ResNet-50 (torchvision `IMAGENET1K_V2`), new 101-way fc | ImageNet-1k | 23.71 M | 4.09 | pretrained model `[MANDATORY]` |
| P2 | EfficientNet-B0 (`IMAGENET1K_V1`), new 101-way head | ImageNet-1k | 4.14 M | 0.38 | modern, compute-efficient pretrained model `[MANDATORY]` per our proposal |
| P3 | ConvNeXt-Tiny (`IMAGENET1K_V1`), new 101-way head | ImageNet-1k | 27.90 M | 4.45 | modern architecture `[OPTIONAL]`, 1 seed |
| E1 | P1 with a frozen backbone (BatchNorm in eval mode), only the fc is trained | ImageNet-1k | 0.21 M of 23.71 M | 4.09 | controlled experiment `[MANDATORY]` |

B0 has 20x fewer parameters than P1 but needs 63 % of its compute. The reason is that the first block runs two convolutions at full 224 x 224 resolution.

### 2.3 Fine-tuning procedure and training setup

| Item | Value |
| --- | --- |
| Optimizer | AdamW, weight decay 0.05 (not on biases or normalisation parameters) |
| Learning rate | P1-P3: new head 1e-3, backbone 1e-4 (discriminative LR, 10x). B0: 1e-3 everywhere. E1: head 1e-3, backbone frozen |
| Schedule | 1 linear warm-up epoch, then cosine decay to 1 % |
| Epochs | P1-P3, E1, E2: 10. B0: 30. Early stopping after 4 epochs without improvement (it never triggered) |
| Batch / precision | batch 64, fp16 automatic mixed precision, gradient clipping 1.0 |
| Checkpoint rule | highest **validation macro-F1** (`best.pt`); test evaluated once per checkpoint |
| Seeds | split seed 42 (fixed); training seeds 42, 123, 2026 for B0, P1, P2, E1, E2; seed 42 for P3 and E3 (E3 subset seed 42) |
| Hardware | Kaggle, NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB), 48 vCPU, data loader with 48 workers. Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0, CUDA 13.0, cuDNN 9.27. Two parallel sessions on the same GPU type, Internet off. |
| Code | commit `a12ccd6` for all 22 runs (`GIT_COMMIT` file in the offline code bundle, recorded in every `env.json`) |

Fine-tuning procedure (P1-P3):
- Replace the ImageNet 1000-way classifier with a randomly initialised 101-way linear layer.
- Train all layers from the ImageNet weights, with the new head at a 10x higher learning rate than the backbone.
- Use exactly the same augmentation, schedule and selection rule as the baseline. The only differences from B0 are the initialisation, the learning-rate split and the epoch count (10 vs 30, fixed in the proposal before any run).

### 2.4 Experimental design

Decision rule (proposal section 4, unchanged): method A is better than B only if both hold:
1. The mean test macro-F1 gap over seeds exceeds the larger of the two seed standard deviations.
2. A McNemar test on seed 42 (same 25,250 test images) gives p < 0.05.

Comparisons with fewer than 2 seeds per arm are labelled *preliminary*.

| ID | Hypothesis (pre-registered) | Changed factor | Fixed | Decision criterion |
| --- | --- | --- | --- | --- |
| E1 `[MANDATORY]` | Full fine-tuning beats a frozen ImageNet backbone by at least 5 pp macro-F1, because ImageNet features under-represent fine food texture. | backbone frozen + BN eval vs all layers trained (P1) | data, split, augmentation (features not cached), schedule, head LR, epochs, seeds | decision rule + gap of at least 5 pp |
| E2 `[OPTIONAL]` | Label smoothing (ε = 0.1) improves macro-F1 by at least 0.5 pp and lowers ECE, because train labels are noisy and test labels are clean. | CE vs CE + label smoothing 0.1 (P1) | everything else | decision rule + at least 0.5 pp + lower ECE |
| E3 `[OPTIONAL]` | With 10 % of the train data, P1 keeps at least 85 % of its full-data macro-F1; B0 does not. | train fraction 10 / 25 / 50 / 100 % (stratified per class) for B0 and P1 | val / test sets, schedule, epochs | ratio of macro-F1 to the 100 % run |

---

## 3. Implementation results

All tables: [outputs/comparison/food101/summary.md](../outputs/comparison/food101/summary.md). Every number traces to a run id, commit, split fingerprint and checkpoint SHA-256 in [runs.csv](../outputs/comparison/food101/runs.csv).

### 3.1 Training behaviour

Figures: [main_curves.png](../outputs/comparison/food101/main_curves.png) and each run's `curves.png`.

| Model (seed 42) | Selected epoch | Val macro-F1 | Train acc / val acc, last epoch | Lowest val loss at epoch | Behaviour |
| --- | ---: | ---: | --- | ---: | --- |
| B0 | 29 / 30 | 67.1 % | 62.7 / 66.3 % | 29 | under-fitting: still improving, train acc below val acc (augmentation + dropout) |
| P1 | 10 / 10 | 82.1 % | 95.7 / 82.0 % | 4 | over-fitting: val loss flat after epoch 4 while train loss falls to 0.16; val F1 still creeps up |
| P2 | 10 / 10 | 80.3 % | 86.6 / 80.3 % | 9 | mild gap, still improving |
| P3 | 10 / 10 | 84.9 % | 97.7 / 84.8 % | 9 | large train-val gap, but val loss still at its minimum |
| E1 | 8 / 10 | 66.0 % | 70.3 / 66.2 % | 10 | converging slowly, no over-fitting (only the fc is trained) |

- **Pretrained models converge fast.** They reach 65-75 % val macro-F1 after one epoch; B0 reaches 6 %.
- **B0 selection.** B0's selected epoch is 28.7 ± 1.5 over seeds, i.e. at the end of training. More epochs would probably help it (limitation).
- **Validation underestimates test.** Test macro-F1 is about 4 pp above validation macro-F1 for every model (P1 82.1 vs 86.1 %, B0 67.1 vs 71.3 %). The validation set is carved from the noisy train list (6.7 % wrong labels), while the test set was cleaned by the authors. The direction and size of the gap fit this explanation. We did not audit validation labels, so it is a hypothesis.
- **Training time per run** (RTX PRO 6000): B0 18.2 min (30 epochs), P1 5.9 min, P2 5.2 min, P3 6.5 min, E1 2.5 min.

### 3.2 Quantitative results (test set, 25,250 images, mean ± sample std over seeds)

| ID | Model | Seeds | Test acc (%) | **Test macro-F1 (%)** | Top-5 (%) | ECE | Macro-F1 w/o 47 dups (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | SimpleCNN (scratch) | 3 | 71.20 ± 0.17 | **71.12 ± 0.17** | 91.46 ± 0.17 | 0.079 ± 0.001 | 71.10 |
| P1 | ResNet-50 | 3 | 85.96 ± 0.11 | **85.99 ± 0.11** | 97.14 ± 0.05 | 0.051 ± 0.001 | 85.98 |
| P2 | EfficientNet-B0 | 3 | 84.79 ± 0.09 | **84.76 ± 0.10** | 96.90 ± 0.06 | 0.019 ± 0.002 | 84.75 |
| P3 * | ConvNeXt-Tiny | 1 | 88.51 | **88.52** | 97.96 | 0.042 | 88.51 |
| E1 | ResNet-50 frozen backbone | 3 | 70.75 ± 0.02 | **70.51 ± 0.05** | 91.38 ± 0.08 | 0.115 ± 0.001 | 70.49 |
| E2 * | ResNet-50 + label smoothing | 3 | 86.37 ± 0.18 | **86.38 ± 0.18** | 97.08 ± 0.02 | 0.104 ± 0.001 | 86.37 |

`*` = optional. The 95 % bootstrap interval for one seed is about ±0.4 pp (P1 seed 42: 85.63-86.46 %), so the seed std (0.1-0.2 pp) is smaller than the test-set sampling noise.

**Method comparison** ([decisions.md](../outputs/comparison/food101/decisions.md), [mcnemar.csv](../outputs/comparison/food101/mcnemar.csv)):

| A vs B | Δ macro-F1 (pp) | max seed std | McNemar (A right only / B right only), p | Verdict |
| --- | ---: | ---: | --- | --- |
| P1 vs B0 | +14.87 | 0.17 | 4,529 / 835, p < 1e-300 | **P1 better** |
| P2 vs B0 | +13.64 | 0.17 | 4,370 / 958, p < 1e-300 | **P2 better** |
| P1 vs P2 | +1.23 | 0.11 | 1,522 / 1,240, p = 9e-8 | **P1 better** |
| P3 vs P1 | +2.53 | n/a (1 seed) | 1,540 / 906, p = 2e-37 | preliminary |
| P3 vs P2 | +3.76 | n/a | 1,773 / 857, p = 3e-71 | preliminary |

ImageNet pretraining is worth about 14-15 pp over the from-scratch baseline. Among the mandatory pretrained models, ResNet-50 beats EfficientNet-B0 by 1.2 pp, but at 10.8x the compute per image (4.09 vs 0.38 GMACs). P2 is also clearly the best calibrated (ECE 0.019). ConvNeXt-Tiny is best overall, but with one seed that is preliminary under our rule. Its McNemar gap (634 more images right than wrong vs P1) is large, though, compared with seed noise of about 0.1 pp.

**ImageNet overlap** ([imagenet_overlap.md](../outputs/comparison/food101/imagenet_overlap.md)):
- Fine-tuned pretrained models score 2.2-2.6 pp higher on the 7 classes with an ImageNet counterpart; B0 shows no such gap (-0.9 pp).
- For the frozen E1 backbone the gap is **+11.8 pp** (81.5 vs 69.7 %). Frozen ImageNet features are good for dishes ImageNet already knows and weak for the rest, and fine-tuning closes most of that gap.
- With 7 classes this is indicative only.

### 3.3 Controlled experiment E1 `[MANDATORY]`: frozen backbone vs full fine-tuning

| | E1 frozen | P1 full fine-tune | Δ |
| --- | ---: | ---: | ---: |
| Test macro-F1 | 70.51 ± 0.05 | 85.99 ± 0.11 | **+15.48 pp** |
| Top-5 | 91.38 | 97.14 | +5.8 pp |
| ECE | 0.115 | 0.051 | |
| Trainable params | 0.21 M | 23.71 M | 115x |
| Train time / run | 2.5 min (4,619 img/s) | 5.9 min (1,936 img/s) | 2.4x |

- **Verdict: hypothesis confirmed.** The gap (15.5 pp) exceeds the 5 pp threshold and the max seed std (0.11 pp), and McNemar gives 4,624 vs 777 discordant images (p < 1e-300).
- **Where fine-tuning helps.** The largest per-class gains are `scallops` +33.7 pp, `huevos_rancheros` +32.4, `gnocchi` +29.9 and `tuna_tartare` +29.5. These are dishes defined by texture or composition, with no ImageNet counterpart. The smallest gains are for visually distinctive or ImageNet-like classes: `edamame` +1.5, `dumplings` +2.2, `hot_dog` +3.3.
- **Calibration.** E1 is under-confident: mean confidence 0.59 against accuracy 0.71.
- **Threat to validity.** E1 kept P1's head learning rate (1e-3) and 10 epochs, so only one factor changes, and its validation F1 was still rising (best epoch 9 ± 1). A probe tuned for its own optimum (higher LR, more epochs) would narrow the gap somewhat. It is very unlikely to close 15 pp, since even the train accuracy of the probe is only 70 %.

### 3.4 Controlled experiment E2 `[OPTIONAL]`: label smoothing

| | P1 (CE) | E2 (CE + LS 0.1) |
| --- | ---: | ---: |
| Test macro-F1 | 85.99 ± 0.11 | 86.38 ± 0.18 (+0.39 pp) |
| ECE | 0.051 | 0.104 |
| Mean confidence - accuracy (seed 42) | +0.050 (over-confident) | −0.104 (under-confident) |
| Predictions with confidence > 0.9 | 77 % | 34 % |

- **Verdict.**
  - The decision rule declares E2 better: the gap 0.39 pp exceeds the max std 0.18, and McNemar gives 882 vs 785 discordant images, p = 0.019.
  - But the pre-registered effect size (at least 0.5 pp) is **not** reached.
  - The calibration half of the hypothesis is **rejected**: ECE doubles.
- **Why ECE rises.** Label smoothing over-corrects. With ε = 0.1 over 101 classes, the target for the true class is 0.901. P1 was over-confident by 5 pp and E2 is under-confident by 10 pp ([calibration.png](../outputs/error_analysis/food101/calibration.png)). ECE measures miscalibration in both directions.
- **Conclusion.** Label smoothing gives a small, statistically detectable accuracy gain on noisy-label training data. It is not a calibration fix at this ε; temperature scaling on validation would be the right tool (not done).

### 3.5 Controlled experiment E3 `[OPTIONAL]`: data efficiency (seed 42 for the subsets)

| Train fraction | B0 macro-F1 | B0 ratio to 100 % | P1 macro-F1 | P1 ratio to 100 % |
| --- | ---: | ---: | ---: | ---: |
| 10 % (6,819 images, 67-68 / class) | 24.92 | 0.35 | 68.03 | 0.79 |
| 25 % | 47.56 | 0.67 | 77.40 | 0.90 |
| 50 % | 61.30 | 0.86 | 82.75 | 0.96 |
| 100 % (3-seed mean) | 71.12 | 1.00 | 85.99 | 1.00 |

- **Verdict: partly rejected.** P1 is far more data-efficient: with 10 % of the data it comes within 3.1 pp of B0 trained on 100 % (68.0 vs 71.1 %), and with 25 % it already exceeds it (77.4 %). It keeps 79 %, not at least 85 %, of its full-data F1, so the quantitative threshold fails. B0 keeps only 35 %.
- **Caveat.** Subset runs use one seed. Each run used the same number of epochs, so smaller fractions also received fewer gradient steps: B0 at 10 % got 3,180 steps vs 31,950 at 100 %. "Less data" therefore also means "less training", which favours the faster-converging pretrained model.
- Figure: [data_efficiency.png](../outputs/comparison/food101/data_efficiency.png).

### 3.6 Compute cost

[compute_cost.md](../outputs/comparison/food101/compute_cost.md). Measured on the RTX PRO 6000. Inference uses fp16 with data already on the GPU.

| Model | Params | GMACs | Train min/run | GPU-h (all seeds) | Train img/s | Infer img/s (bs 128) | Latency bs 1 (ms) | Checkpoint (MB) | Macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1.20 M | 2.59 | 18.2 | 0.91 | 1,903 | 11,490 | 0.31 | 5 | 71.12 |
| P1 | 23.71 M | 4.09 | 5.9 | 0.29 | 1,936 | 7,617 | 1.57 | 91 | 85.99 |
| P2 | 4.14 M | 0.38 | 5.2 | 0.26 | 2,171 | 14,512 | 2.10 | 16 | 84.76 |
| P3 * | 27.90 M | 4.45 | 6.5 | 0.11 | 1,745 | 7,249 | 1.56 | 106 | 88.52 |
| E1 | 0.21 M trainable | 4.09 | 2.5 | 0.12 | 4,619 | 7,567 | 1.79 | 91 | 70.51 |
| E2 * | 23.71 M | 4.09 | 6.0 | 0.30 | 1,915 | 7,618 | 1.60 | 91 | 86.38 |

**Total measured training time for all 22 runs: 2.4 GPU-hours.** The proposal estimated about 14 T4 GPU-hours; the faster GPU made the difference.

Observations:
- **Training is bound by data loading, not by the model.** B0, P1 and P2 all train at about 1,900-2,200 img/s despite a 10x spread in GMACs. JPEG decoding and augmentation on 48 CPU workers is the limit.
  - E1 is faster (4,619 img/s) because nothing is back-propagated through the backbone, and the data loader is no longer the only bottleneck.
  - B0 is the *most expensive* model to train in total (0.91 GPU-h): it needs 30 epochs instead of 10.
- **GMACs do not predict latency.**
  - At batch 128, P2 (0.38 GMACs) is 1.9x faster than P1 (4.09 GMACs), as expected.
  - At batch 1, P2 is *slower* than P1: 2.10 vs 1.57 ms. Its many small depthwise layers make single-image inference bound by kernel-launch overhead, not arithmetic.
  - For a single-image service, P1 or P3 gives better F1 at lower latency. For batch throughput or a 16 MB model, P2 is the better choice.
- **Hardware constraints.** All runs fit easily in 95 GB of GPU memory. On the proposal's T4 (16 GB), batch 64 at 224 px fits for every model, but the fixed 10-epoch budget was chosen with T4 throughput in mind.

### 3.7 Error analysis

#### Taxonomy

Every test error of a model goes into exactly one bucket ([taxonomy.md](../outputs/error_analysis/food101/taxonomy.md)). Seed 42 for every model.
- **Consensus:** P1, P2 and P3 all predict the same wrong class. No single model is to blame; this suggests a label problem or a genuinely ambiguous photo.
- **Sibling:** the true and predicted class are in the same culinary group. There are 13 groups, defined from the class list alone in `src/data/food_groups.py`, before looking at confusions.
- **Near-miss:** a different group, but the true class is in the top 5.
- **Gross:** everything else.

| Model | Errors (rate) | Consensus | Sibling | Near-miss | Gross | Confident (> 0.9) errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 7,229 (28.6 %) | 4.8 % | 34.0 % | 38.6 % | 22.5 % | 165 (2.3 %) |
| P1 | 3,535 (14.0 %) | 19.1 % | 32.6 % | 35.8 % | 12.5 % | 731 (20.7 %) |
| P2 | 3,817 (15.1 %) | 17.7 % | 31.5 % | 37.6 % | 13.3 % | 411 (10.8 %) |
| P3 | 2,901 (11.5 %) | 23.2 % | 33.0 % | 33.2 % | 10.6 % | 601 (20.7 %) |

1. **Fine-grained confusions dominate.** About half of every pretrained model's errors are consensus or sibling errors. The top confusions are the same for all models:
   - P1: `filet_mignon -> steak` (45, 18 % of the class), `chocolate_mousse -> chocolate_cake` (28), `steak -> prime_rib` (25), `beef_tartare -> tuna_tartare` (21).
   - The worst classes for every model are `steak` (P3 F1 0.62), `pork_chop`, `filet_mignon` and `chocolate_mousse`: the meat-cut and chocolate-dessert groups ([group_confusion_p3_convnext_tiny.png](../outputs/error_analysis/food101/group_confusion_p3_convnext_tiny.png)).
   - Cause: these dishes differ in cut or recipe, not in appearance, and the training labels are noisiest exactly there (the audit found 4 of 10 `filet_mignon` train labels wrong).
2. **Better models' errors look more like consensus.** The consensus share grows from P2 (17.7 %) to P3 (23.2 %). As a model gets better, a larger part of what it still gets wrong is images that no model gets right.
   - 674 test images (2.7 %) get the same wrong class from all three pretrained models.
   - 1,421 images (5.6 %) are wrong for all four models.
   - The oracle accuracy (at least one model right) is 94.4 %.
3. **Pretrained models fail confidently.** 20.7 % of P1's and P3's errors have confidence above 0.9, against 2.3 % for B0 ([calibration.png](../outputs/error_analysis/food101/calibration.png)), and 71-75 % of those confident errors are consensus or sibling errors. P2 is better calibrated (10.8 % confident errors).
4. **Leakage does not explain anything.** Only 2-8 errors per model involve the 47 flagged near-duplicate test images.

#### Manual review of P1 errors

[review_result.md](../outputs/error_analysis/food101/review_result.md); verdicts in [data/error_review/p1_error_review.csv](../data/error_review/p1_error_review.csv).

Protocol:
- **Sample.** 125 errors of the M2 P1 seed-42 checkpoint: 75 random errors (seed 42) and the 50 most confident remaining errors. Shown on review sheets as the eval-time centre crop with label, prediction and confidence.
- **Verdicts.** `label_wrong` (the photo shows the predicted or another dish), `ambiguous` (several dishes, dish not identifiable, or both labels plausible) or `model_error` (the photo clearly shows the labelled dish).
- **Rater.** One rater judged all 125 images. A food-identification judgement from one rater is subjective, so treat the rates as rough estimates.

| Stratum | n | Label wrong | Ambiguous | Model error |
| --- | ---: | --- | --- | --- |
| Random errors | 75 | 2 (3 %, CI 1-9) | 28 (37 %, CI 27-49) | 45 (60 %, CI 49-70) |
| 50 most confident errors | 50 | 18 (36 %, CI 24-50) | 23 (46 %, CI 33-60) | 9 (18 %, CI 10-31) |

- **A typical error is a real model error, often on a hard photo.** About 60 % of random errors are model errors and 37 % are ambiguous.
- **A *confident* error is usually not the model's fault.** Of the 50 most confident errors, only 18 % are clear model errors. Examples of wrong test labels: nigiri labelled `sashimi`; oysters labelled `scallops` and `mussels`; a burger with fries labelled `lobster_roll_sandwich`; a tomato-topped toast labelled `garlic_bread`. So the test set, described as manually cleaned, still contains wrong labels. A noticeable part of the confident "errors" are correct predictions.
- **The verdicts are consistent with the final models.** The review sample comes from the M2 checkpoint (run `food101-p1_resnet50-s42-20261029-195643`, run folder in git at commit `e39b497`). We checked which reviewed images the final seed-42 models still get wrong:

  | Verdict | n | still wrong for P1 / P2 / P3 | wrong for all three, same class as before |
  | --- | ---: | --- | ---: |
  | label_wrong | 20 | 100 % / 90 % / 95 % | 90 % |
  | ambiguous | 51 | 96 % / 80 % / 71 % | 41 % |
  | model_error | 54 | 78 % / 57 % / 56 % | 24 % |

  Images judged "label wrong" fool every model in the same way. Images judged "model error" are often fixed by another architecture. That is the pattern expected if the verdicts are right.
- **Rough estimate.** About 3 % of P1's errors (about 0.4 % of the test set) are wrong test labels, and about a third involve ambiguous photos. The reachable ceiling for this metric is therefore below 100 % by at least a few tenths of a percent, and macro-F1 differences of the size of E2's (0.4 pp) are close to the label-noise level.

#### Qualitative results and interpretability

**Prediction grids per run** (`outputs/runs/food101/<experiment>/seed42/eval/`):
- `examples_correct_random.png`, `examples_correct_low_confidence.png`
- `examples_incorrect_random.png`, `examples_incorrect_high_confidence.png`
- `confusion_matrix_normalized.png`

**Grad-CAM `[OPTIONAL]`** ([gradcam.png](../outputs/error_analysis/food101/gradcam.png); 12 P1 errors, 3 per bucket, plus 4 correct predictions; all four models):
- **On consensus errors, all models attend to the same, plausible evidence.** On the "scallops" photo they all look at the oyster shells; on the "garlic_bread" photo, at the tomato topping that defines bruschetta. This supports the label-problem reading.
- **On sibling errors the attention lands on the dish itself**, e.g. the cake slice in cheesecake -> tiramisu. The confusion is in what the visual evidence means, not in where the model looks.
- **B0's maps are more scattered** and often cover background or plate texture. This fits its lower accuracy and its errors spreading over unrelated groups (22.5 % gross errors vs 10.6 % for P3).
- Grad-CAM shows where a decision is sensitive. It does not prove *why* the model decided, and 16 images are an illustration, not a measurement.

### 3.8 Limitations

- **Fixed protocol, no tuning.** The protocol was not tuned. B0 was still improving at epoch 30; E1's probe kept P1's learning rate and was still rising at epoch 10. Both comparisons with B0 and E1 are therefore *at a fixed budget*, not at each method's optimum.
- **One seed for the optional runs.** P3 and E3 use one seed, so their comparisons are preliminary.
- **Label noise.**
  - Train labels are noisy (6.7 %, CI 4.0-11.2 %), validation inherits that noise, and we found wrong labels in the test set too.
  - The audit (200 train images) and the error review (125 test errors) each have a single rater, so we have no inter-rater agreement.
  - We did not audit the validation labels.
- **Leakage.** The split unit is the image. Same-uploader photos that are not pHash-similar cannot be detected.
- **Timing and hardware.** Timings are for one GPU type (RTX PRO 6000) with 48 CPU workers. Training throughput is bound by data loading, so on another machine the relative training costs can change. M2 numbers (RTX 3060) are superseded; for example, the P1 seed-42 test macro-F1 moved from 86.38 % (M2) to 86.05 % (M3) with a different GPU and library stack, which is within the seed spread plus run-to-run GPU non-determinism.
- **E3 confound.** E3 confounds data size with the number of gradient steps (section 3.5).
- **Calibration.** Calibration was measured but not corrected (no temperature scaling).

### 3.9 Conclusions

1. **Pretraining.** ImageNet pretraining adds about 14-15 pp test macro-F1 over a from-scratch CNN on Food-101, and it converges in a third of the epochs. Seed std is about 0.1-0.2 pp; McNemar p < 1e-300.
2. **Full fine-tuning matters** (E1, mandatory). Training all layers adds 15.5 pp over a frozen backbone. The gain is largest on texture- and composition-defined dishes that ImageNet does not contain.
3. **Choice of pretrained model** depends on the constraint:
   - ConvNeXt-Tiny gives the best F1 (88.5 %, one seed).
   - ResNet-50 beats EfficientNet-B0 by 1.2 pp at 10.8x the GMACs.
   - EfficientNet-B0 is the best calibrated and has the highest batch throughput, but the highest batch-1 latency.
4. **Label smoothing** (E2) gives +0.4 pp but makes calibration worse (it over-corrects to under-confidence). **Pretraining is data-efficient** (E3): P1 at 25 % data beats B0 at 100 %, but at 10 % it keeps 79 %, not 85 %, of its F1.
5. **What remains is mostly fine-grained ambiguity and label noise.** About half of the pretrained models' errors fall within one culinary group or are shared by all models. Among the most confident errors, wrong or ambiguous labels outnumber clear model errors about 4 to 1.

---

## 4. Deviations from the proposal

| Item | Proposal | Final |
| --- | --- | --- |
| Val / train size | 7,575 / 68,175 | 7,561 / 68,189 (14 near-duplicate val images moved to train) |
| Hardware | Kaggle T4 (16 GB) | Kaggle RTX PRO 6000 Blackwell (95 GB), Internet off; same code, configs and split |
| Compute | about 14 T4 GPU-h estimated | 2.4 GPU-h measured |
| B0 compute | about 1 GFLOP | 2.59 GMACs measured |
| M2 runs | - | M2 seed-42 runs (RTX 3060) superseded by M3 reruns so that all timings come from one GPU |
| Error analysis | taxonomy | taxonomy + manual review of 125 test errors + Grad-CAM (optional) |

Everything else (models, augmentation, optimiser, schedule, epochs, seeds, metrics, decision rule, hypotheses) is exactly as pre-registered.

## 5. Reproduction

```bash
pip install -r requirements.txt                                   # exact local versions: requirements-lock.txt
python scripts/download_data.py && python scripts/preresize.py    # split lists are committed (data/splits/)
# training: notebooks/kaggle_train.ipynb (Internet off; inputs from scripts/make_kaggle_bundle.py) or locally:
python scripts/run_experiments.py --models b0_simple_cnn p1_resnet50 p2_efficientnet_b0 e1_resnet50_linear_probe \
    e2_resnet50_label_smoothing --seeds 42 123 2026
python scripts/run_experiments.py --models p3_convnext_tiny e3_b0_simple_cnn_frac10 e3_b0_simple_cnn_frac25 \
    e3_b0_simple_cnn_frac50 e3_p1_resnet50_frac10 e3_p1_resnet50_frac25 e3_p1_resnet50_frac50 --seeds 42
python scripts/compare.py
python scripts/error_analysis.py taxonomy && python scripts/error_analysis.py review-score
python scripts/gradcam.py --models b0_simple_cnn p1_resnet50 p2_efficientnet_b0 p3_convnext_tiny \
    --set data.image_root=data/resized/food-101-256
```

Checkpoints: `checkpoints_food101.zip` (`[TODO: link]`) holds `best.pt` of all 22 runs in the repository layout plus `CHECKPOINTS_SHA256.txt`; unzip at the repository root (README section 7).

## AI usage disclosure

[To be completed by the group.]
