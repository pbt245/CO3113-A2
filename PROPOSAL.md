# Assignment 2 - Dataset Proposal (Milestone M1)

**Course:** Deep Learning and Its Applications (CO3133), Semester 261 - Instructor: Lê Thành Sách
**Group:** G-roundhog - Phan Bá Thanh (2353084), Trần Công Hoàng Phước (2352966)
**Track:** Image classification
**Submitted:** 2026-10-07

## Approval record

| Field | Value |
| --- | --- |
| Status | **Pending review** |
| Decision (Approved / Approved with conditions / Rejected) | - |
| Conditions from instructor | - |
| Decision date | - |

Main implementation work is blocked until this proposal is approved .

---

## 1. Task formulation

Single-label, 101-way classification of dish photographs: given one RGB photo of a dish,
predict which of 101 food categories it shows.

- **Input:** one RGB image, resized for the model (224 × 224 for pretrained models).
- **Output:** logits over 101 classes, trained with cross-entropy.
- **Why it is a "specialized" task:** fine-grained categories with high intra-class variance
  (the same dish photographed with different plating, lighting and framing) and low
  inter-class variance (e.g. `steak` / `filet_mignon` / `pork_chop`, `ramen` / `pho`).
  The training labels are known to contain noise , so the task also tests
  robustness to imperfect supervision.

## 2. Dataset

### 2.1 Source

**Food-101** - L. Bossard, M. Guillaumin, L. Van Gool, *"Food-101 – Mining Discriminative
Components with Random Forests"*, ECCV 2014.

- Official page: <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>
- Also distributed as `torchvision.datasets.Food101` and on Hugging Face as `ethz/food101`;
  we will use the torchvision loader with the official archive so split files match the authors'.
- Images were collected from foodspotting.com. The authors distribute them for research use;
  we will not redistribute images (only code, split index files and checkpoints).

### 2.2 Size and threshold check

| Quantity | Value | Spec threshold  | Met? |
| --- | ---: | ---: | :---: |
| Classes | 101 | ≥ 5 | Yes |
| Total images | 101,000 (1,000 per class) | - | - |
| Official train images | 75,750 (750 per class) | - | - |
| Our training images after val hold-out | **68,175** (675 per class) | ≥ 5,000 | Yes |
| Official test images | 25,250 (250 per class) | - | - |
| Archive size | ≈ 5 GB (`food-101.tar.gz`) | - | - |

The dataset is not MNIST / Fashion-MNIST and is ≈ 30× larger per image than the A1 data
(images up to 512 px vs 28 px).

### 2.3 Known properties and biases (to be quantified in the M2 EDA)

1. **Class balance:** exactly balanced by construction (750 train / 250 test per class),
   so imbalance is not an issue; macro-F1 and accuracy will be close, but both are reported.
2. **Label noise in train only:** the authors state that test images were manually reviewed,
   while training images were left uncleaned on purpose and contain some wrong labels and
   intense colour artefacts. Train and test therefore differ in label quality.
3. **Resolution:** images were rescaled by the authors to a maximum side length of 512 px;
   aspect ratios vary. The EDA will report the width/height/aspect-ratio distribution.
4. **Cultural / geographic bias:** categories and photos come from a mostly
   North-American/European photo-sharing site (e.g. many Western desserts and
   restaurant dishes; very few Vietnamese dishes - `pho` is one of the few). Results do not
   transfer to "food in general".
5. **Overlap with ImageNet pretraining:** ImageNet-1k contains some food classes
   (e.g. pizza, cheeseburger, hotdog, ice cream, carbonara, guacamole). Pretrained models
   therefore start with partial food knowledge; we will report per-class F1 split into
   "has an ImageNet counterpart" vs "does not" to make this visible.

### 2.4 Planned EDA (M2, task-appropriate for classification)

- Per-class counts per split (train / val / test) - verifies the balance claim.
- Image width, height and aspect-ratio histograms; file-size distribution.
- Per-channel mean / std and brightness distribution (also used for normalisation sanity check).
- Sample grid per class and grids for the most similar class pairs.
- Label-noise audit: manual inspection of a random sample of 10 train images per class
  for 20 randomly chosen classes (200 images); report the observed wrong-label rate with a
  95 % confidence interval.
- Near-duplicate check across splits .

## 3. Splitting plan

### 3.1 Splits

| Split | Source | Size | Per class |
| --- | --- | ---: | ---: |
| Train | official train minus val | 68,175 | 675 |
| Validation | stratified 10 % of official train | 7,575 | 75 |
| Test | official test (untouched) | 25,250 | 250 |

- **Split unit:** image. Food-101 provides no uploader / restaurant / dish-instance ID,
  so grouping by a higher-level unit is not possible (listed as a limitation).
- **Stratification:** `sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)`
  on the official train list, stratified by class.
- **Seed:** split seed fixed at **42** (same convention as A1). Training seeds: 42, 123, 2026.
- The split is written once to `data/splits/{train,val,test}.txt` (one relative path per line)
  and committed, so every model uses byte-identical splits.

### 3.2 Use of each split

- Train: gradient updates only.
- Validation: hyper-parameter choices, early stopping and checkpoint selection
  (best validation macro-F1).
- Test: evaluated **once per final checkpoint**, never used for any decision.

### 3.3 Leakage prevention

- The test split is the authors' official, manually reviewed list; no test image enters
  training or model selection.
- **Near-duplicate check:** compute a 64-bit perceptual hash (pHash) for every image and flag
  cross-split pairs with Hamming distance ≤ 4. Flagged val images are moved to train; flagged
  test images are kept (to stay comparable with published numbers) but we additionally report
  test metrics with them excluded, and report how many there were.
- Normalisation statistics: ImageNet mean/std for pretrained models (no statistics computed
  on test); for the from-scratch baseline, mean/std computed on the train split only.

## 4. Metrics

| Metric | Role |
| --- | --- |
| Top-1 accuracy | Mandatory  |
| Macro-F1 | Mandatory  - primary model-selection metric |
| Top-5 accuracy | Secondary, standard for Food-101 |
| Per-class F1 + confusion matrix (top confused pairs) | Error analysis |
| Parameter count, training time (GPU-h), inference throughput (img/s, batch 128) and latency (ms, batch 1) | Compute cost  |

**Decision rule (same as A1):** method A is declared better than B only if the difference in
mean test macro-F1 over 3 seeds exceeds the larger of the two seed standard deviations
**and** a McNemar test on seed 42 rejects equal error rates at p < 0.05.

## 5. Model plan

Models are chosen to fit a Kaggle T4 (16 GB, fp16 tensor cores, no bf16) within the weekly quota.

| ID | Model | Init | Params (≈) | GFLOPs @224 | Role | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| B0 | SimpleCNN: 4 blocks × (2 × conv3×3–BN–ReLU) + max-pool, channels 32-64-128-256, global avg-pool, dropout 0.3, linear head | random | 1.2 M | ≈ 1 | Simple baseline | `[MANDATORY]` |
| P1 | ResNet-50 (torchvision `IMAGENET1K_V2`), new 101-way head | ImageNet-1k | 23.7 M | 4.1 | Pretrained model | `[MANDATORY]` |
| P2 | EfficientNet-B0 (torchvision `IMAGENET1K_V1`), new 101-way head | ImageNet-1k | 4.1 M | 0.39 | Modern, compute-efficient pretrained model | `[MANDATORY]` (we commit to ≥ 1, plan 2) |
| P3 | ConvNeXt-Tiny (torchvision `IMAGENET1K_V1`) | ImageNet-1k | 27.9 M | 4.5 | Modern architecture comparison | `[OPTIONAL]`, 1 seed |

Exact parameter counts will be printed by the code and reported in M2.

### 5.1 Fine-tuning procedure (initial plan; tuned on validation only)

- Data: images pre-resized once to shorter side 256 px (CPU-only Kaggle notebook, which does
  not consume GPU quota) and stored as a private Kaggle dataset, so each GPU session skips the
  5 GB download and most JPEG decode cost.
- Input: `RandomResizedCrop(224, scale=(0.35, 1.0))` + horizontal flip + light colour jitter
  for training; `Resize(256)` + `CenterCrop(224)` for val/test.
- Optimiser: AdamW, weight decay 0.05; discriminative LR - head 1e-3, backbone 1e-4.
- Schedule: 1 epoch linear warm-up, then cosine decay; **10 epochs**; batch size 64;
  fp16 automatic mixed precision with gradient scaling.
- B0 (from scratch): same pipeline, AdamW LR 1e-3, 30 epochs.
- Checkpoint selection: best validation macro-F1; early stop if no improvement for 4 epochs.
  A checkpoint is saved every epoch so runs can resume across the 12 h session limit.

### 5.2 Controlled experiments / ablations

Each experiment changes **one** factor; split, resolution, augmentation, epochs, optimiser and
seeds stay fixed.

| ID | Hypothesis | Changed factor | Fixed | Decision criterion | Status |
| --- | --- | --- | --- | --- | --- |
| E1 | Full fine-tuning beats a frozen backbone (linear probe) by ≥ 5 pp macro-F1, because ImageNet features under-represent fine food texture. | Frozen backbone (BN in eval mode) + trained head vs all layers trained (P1). Same augmentation in both arms; features are **not** cached, so augmentation is not a confound. | everything else | Listed in section 4 | `[MANDATORY]` |
| E2 | Label smoothing (ε = 0.1) improves test macro-F1 by ≥ 0.5 pp and lowers calibration error (ECE), because train labels are noisy while test labels are clean. | CE vs CE + label smoothing (P1) | everything else | Listed in section 4 + ECE | `[OPTIONAL]` |
| E3 | Pretrained models are far more data-efficient: with 10 % of train data P1 retains ≥ 85 % of its full-data macro-F1, B0 does not. | Train fraction 10 / 25 / 50 / 100 % (stratified, seed 42) for B0 and P1 | val/test sets | ratio of macro-F1 vs 100 % | `[OPTIONAL]` |

## 6. Compute estimate

**Hardware:** Kaggle notebook, NVIDIA Tesla T4 16 GB (P100 16 GB as fallback), 4 vCPUs;
PyTorch 2.x + CUDA. Limits: 12 h per session, 30 GPU-hours per week per account
(the group has two accounts).

**Assumptions (to be replaced by a measured 1-epoch timing run before any full run):**
fp16 training throughput on a T4 at 224 px of ≈ 250 img/s for ResNet-50, ≈ 450 img/s for
EfficientNet-B0, ≈ 150 img/s for ConvNeXt-T, and ≈ 700 img/s for B0 (data-loader-bound).
With 68,175 training images plus validation, this gives the per-epoch budgets below.

| Runs | min/epoch | Epochs | Seeds | Est. GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| 1-epoch timing runs for all models | - | 1 | 1 | 0.5 |
| B0 SimpleCNN from scratch | 2 | 30 | 3 | 3.0 |
| P1 ResNet-50 full fine-tune | 5 | 10 | 3 | 2.5 |
| P2 EfficientNet-B0 full fine-tune | 3 | 10 | 3 | 1.5 |
| E1 linear probe on P1 (frozen backbone, no backward through it) | 2 | 10 | 3 | 1.0 |
| **Mandatory total** | | | | **≈ 8.5** |
| E2 label smoothing on P1 (optional) | 5 | 10 | 3 | 2.5 |
| E3 data-fraction sweep, 10/25/50 % for B0 + P1 (optional) | - | - | 1 | 1.6 |
| P3 ConvNeXt-Tiny (optional) | 8 | 10 | 1 | 1.3 |
| **Total including optional** | | | | **≈ 14** |

Pre-resizing and the pHash duplicate check run in CPU-only notebooks (≈ 1 h, no GPU quota).
The total fits within one account's weekly quota with ≈ 2× headroom for estimation error and
re-runs; no single run exceeds ≈ 1 h, well under the 12 h session limit.

Storage: ≈ 5 GB raw archive (read from Kaggle input, not copied), ≈ 2 GB after pre-resizing;
checkpoints ≈ 95 MB (ResNet-50), ≈ 17 MB (EfficientNet-B0), ≈ 110 MB (ConvNeXt-T) - only the
selected checkpoint per configuration is kept.

**Fallback if measured throughput is lower than assumed:** run B0 with 3 seeds × 20 epochs,
then drop optional items in the order P3 → E3 → E2.

## 7. Deliverables timeline

| Milestone | Content |
| --- | --- |
| M1 (this document) | Dataset proposal: task, data, split, metrics, baseline, compute estimate |
| M2 | Split files + EDA complete; B0 and P1 trained with preliminary metrics |
| M3 | P2, E1 (and optional E2/E3/P3), error taxonomy, compute-cost analysis, report, slides, video, pages, checkpoints |

## 8. Risks and limitations

- Split unit is the image (no uploader IDs), so same-user near-duplicates may remain across
  train/test even after the pHash check.
- Train-label noise makes the train/val loss curves noisier than the test behaviour.
- ImageNet overlap means pretrained-model gains partly reflect prior exposure to food
  categories, not only transfer of generic features.
- T4 compute limits the plan to 224 px inputs, 10 fine-tuning epochs and mid-sized backbones;
  larger models (e.g. ViT-B/16) are out of scope. Kaggle quota may force fewer B0 epochs.
