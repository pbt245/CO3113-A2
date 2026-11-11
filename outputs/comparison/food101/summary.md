# Comparison - food101

Mean ± sample std over training seeds (same split for every run). Checkpoint = best validation macro-F1; the test set is evaluated once per checkpoint. `*` = optional experiment. "w/o dups" excludes the test images flagged as pHash near-duplicates of train/val. Hardware: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']; inference timing fp16 on CUDA = [True].

## Main models

| ID | Experiment | Seeds | Params (M) | GMACs | Test acc (%) | Test macro-F1 (%) | Top-5 (%) | ECE | Macro-F1 w/o dups (%) | Best epoch | Train min/epoch | Train img/s | Infer img/s (bs128) | Latency bs1 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | b0_simple_cnn | 42 123 2026 | 1.20 | 2.59 | 71.20 ± 0.17 | 71.12 ± 0.17 | 91.46 ± 0.17 | 0.079 ± 0.001 | 71.10 ± 0.18 | 28.7 ± 1.5 | 0.6 ± 0.0 | 1903 ± 24 | 11490 ± 1 | 0.31 ± 0.00 |
| P1 | p1_resnet50 | 42 123 2026 | 23.71 | 4.09 | 85.96 ± 0.11 | 85.99 ± 0.11 | 97.14 ± 0.05 | 0.051 ± 0.001 | 85.98 ± 0.11 | 10.0 ± 0.0 | 0.6 ± 0.0 | 1936 ± 3 | 7617 ± 3 | 1.57 ± 0.01 |
| P2 | p2_efficientnet_b0 | 42 123 2026 | 4.14 | 0.38 | 84.79 ± 0.09 | 84.76 ± 0.10 | 96.90 ± 0.06 | 0.019 ± 0.002 | 84.75 ± 0.10 | 9.0 ± 1.0 | 0.5 ± 0.0 | 2171 ± 14 | 14512 ± 7 | 2.10 ± 0.01 |
| P3 | p3_convnext_tiny * | 42 | 27.90 | 4.45 | 88.51 | 88.52 | 97.96 | 0.042 | 88.51 | 10.0 | 0.7 | 1745 | 7249 | 1.56 |

## Ablations / controlled experiments

| ID | Experiment | Seeds | Params (M) | GMACs | Test acc (%) | Test macro-F1 (%) | Top-5 (%) | ECE | Macro-F1 w/o dups (%) | Best epoch | Train min/epoch | Train img/s | Infer img/s (bs128) | Latency bs1 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | e1_resnet50_linear_probe | 42 123 2026 | 23.71 | 4.09 | 70.75 ± 0.02 | 70.51 ± 0.05 | 91.38 ± 0.08 | 0.115 ± 0.001 | 70.49 ± 0.05 | 9.0 ± 1.0 | 0.2 ± 0.0 | 4619 ± 17 | 7567 ± 5 | 1.79 ± 0.01 |
| E2 | e2_resnet50_label_smoothing * | 42 123 2026 | 23.71 | 4.09 | 86.37 ± 0.18 | 86.38 ± 0.18 | 97.08 ± 0.02 | 0.104 ± 0.001 | 86.37 ± 0.18 | 8.7 ± 0.6 | 0.6 ± 0.0 | 1915 ± 30 | 7618 ± 2 | 1.60 ± 0.02 |
| E3 | e3_b0_simple_cnn_frac10 * | 42 | 1.20 | 2.59 | 27.85 | 24.92 | 57.33 | 0.090 | 24.93 | 30.0 | 0.1 | 1399 | 11493 | 0.31 |
| E3 | e3_b0_simple_cnn_frac25 * | 42 | 1.20 | 2.59 | 48.67 | 47.56 | 77.32 | 0.128 | 47.57 | 30.0 | 0.2 | 1586 | 11496 | 0.32 |
| E3 | e3_b0_simple_cnn_frac50 * | 42 | 1.20 | 2.59 | 61.57 | 61.30 | 86.40 | 0.101 | 61.30 | 30.0 | 0.3 | 1795 | 11490 | 0.31 |
| E3 | e3_p1_resnet50_frac10 * | 42 | 23.71 | 4.09 | 68.04 | 68.03 | 89.49 | 0.058 | 68.01 | 8.0 | 0.1 | 1376 | 7621 | 1.60 |
| E3 | e3_p1_resnet50_frac25 * | 42 | 23.71 | 4.09 | 77.41 | 77.40 | 93.87 | 0.062 | 77.40 | 9.0 | 0.2 | 1585 | 7621 | 1.61 |
| E3 | e3_p1_resnet50_frac50 * | 42 | 23.71 | 4.09 | 82.78 | 82.75 | 96.07 | 0.056 | 82.74 | 10.0 | 0.3 | 1804 | 7622 | 1.59 |

## Traceability

Commits used: ['a12ccd6e4bb5a1b0ac38c7171abb946404018199']
Train split fingerprints: ['8f130b6f7c76d577']
