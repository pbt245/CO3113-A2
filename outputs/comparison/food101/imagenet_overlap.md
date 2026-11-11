# Macro-F1 on classes with / without an ImageNet-1k counterpart

Classes with a counterpart: see src/data/imagenet_overlap.py (7 of 101).

| Experiment | With counterpart (%) | Without (%) | Gap (pp) |
|---|---|---|---|
| b0_simple_cnn | 70.27 ± 0.98 | 71.18 ± 0.19 | -0.91 |
| p1_resnet50 | 88.05 ± 0.15 | 85.84 ± 0.12 | +2.21 |
| p2_efficientnet_b0 | 87.13 ± 0.19 | 84.58 ± 0.12 | +2.55 |
| p3_convnext_tiny | 90.69 | 88.36 | +2.33 |
| e1_resnet50_linear_probe | 81.45 ± 0.09 | 69.69 ± 0.05 | +11.76 |
| e2_resnet50_label_smoothing | 88.39 ± 0.40 | 86.23 ± 0.16 | +2.17 |
| e3_b0_simple_cnn_frac10 | 19.31 | 25.34 | -6.03 |
| e3_b0_simple_cnn_frac25 | 43.56 | 47.86 | -4.30 |
| e3_b0_simple_cnn_frac50 | 59.38 | 61.45 | -2.06 |
| e3_p1_resnet50_frac10 | 73.18 | 67.64 | +5.53 |
| e3_p1_resnet50_frac25 | 80.82 | 77.15 | +3.67 |
| e3_p1_resnet50_frac50 | 85.59 | 82.53 | +3.06 |
