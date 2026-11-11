# Compute cost - measured

Hardware: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']. Training: fp16 AMP, batch 64, data loading included. Inference: fp16, forward pass only on data already on the GPU (throughput at batch 128, latency at batch 1, mean of 200 images). Test pass = full 25,250-image evaluation incl. JPEG decoding. Checkpoint = `best.pt` (model weights + metadata, no optimizer state). `*` = optional.

| ID | Experiment | Seeds | Params trainable / total (M) | GMACs | Epochs | Min/epoch | Train min/seed | GPU-h (all seeds) | Train img/s | Infer img/s (bs128) | Latency bs1 (ms) | Test pass (s) | Checkpoint (MB) | Test macro-F1 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | b0_simple_cnn | 3 | 1.20 / 1.20 | 2.59 | 30 | 0.6 | 18.2 | 0.91 | 1903 | 11490 | 0.31 | 30 | 5 | 71.12 ± 0.17 |
| P1 | p1_resnet50 | 3 | 23.71 / 23.71 | 4.09 | 10 | 0.6 | 5.9 | 0.29 | 1936 | 7617 | 1.57 | 10 | 91 | 85.99 ± 0.11 |
| P2 | p2_efficientnet_b0 | 3 | 4.14 / 4.14 | 0.38 | 10 | 0.5 | 5.2 | 0.26 | 2171 | 14512 | 2.10 | 9 | 16 | 84.76 ± 0.10 |
| P3 | p3_convnext_tiny * | 1 | 27.90 / 27.90 | 4.45 | 10 | 0.7 | 6.5 | 0.11 | 1745 | 7249 | 1.56 | 10 | 106 | 88.52 |
| E1 | e1_resnet50_linear_probe | 3 | 0.21 / 23.71 | 4.09 | 10 | 0.2 | 2.5 | 0.12 | 4619 | 7567 | 1.79 | 10 | 91 | 70.51 ± 0.05 |
| E2 | e2_resnet50_label_smoothing * | 3 | 23.71 / 23.71 | 4.09 | 10 | 0.6 | 6.0 | 0.30 | 1915 | 7618 | 1.60 | 15 | 91 | 86.38 ± 0.18 |
| E3 | e3_b0_simple_cnn_frac10 * | 1 | 1.20 / 1.20 | 2.59 | 30 | 0.1 | 2.5 | 0.04 | 1399 | 11493 | 0.31 | 8 | 5 | 24.92 |
| E3 | e3_b0_simple_cnn_frac25 * | 1 | 1.20 / 1.20 | 2.59 | 30 | 0.2 | 5.4 | 0.09 | 1586 | 11496 | 0.32 | 8 | 5 | 47.56 |
| E3 | e3_b0_simple_cnn_frac50 * | 1 | 1.20 / 1.20 | 2.59 | 30 | 0.3 | 9.5 | 0.16 | 1795 | 11490 | 0.31 | 8 | 5 | 61.30 |
| E3 | e3_p1_resnet50_frac10 * | 1 | 23.71 / 23.71 | 4.09 | 10 | 0.1 | 0.8 | 0.01 | 1376 | 7621 | 1.60 | 10 | 91 | 68.03 |
| E3 | e3_p1_resnet50_frac25 * | 1 | 23.71 / 23.71 | 4.09 | 10 | 0.2 | 1.8 | 0.03 | 1585 | 7621 | 1.61 | 10 | 91 | 77.40 |
| E3 | e3_p1_resnet50_frac50 * | 1 | 23.71 / 23.71 | 4.09 | 10 | 0.3 | 3.2 | 0.05 | 1804 | 7622 | 1.59 | 10 | 91 | 82.75 |

**Total measured training time for all runs above: 2.4 GPU-hours.**
