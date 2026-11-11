# Decisions (proposal section 4 rule)

A is declared better than B only if |mean test macro-F1 gap| over seeds exceeds the larger seed std **and** McNemar on seed 42 gives p < 0.05.

| A | B | Δ macro-F1 A−B (pp) | max std (pp) | McNemar p | gap > std | p < 0.05 | Verdict |
|---|---|---|---|---|---|---|---|
| b0_simple_cnn | p1_resnet50 | -14.87 | 0.17 | 0 | yes | yes | **p1_resnet50 better** |
| b0_simple_cnn | p2_efficientnet_b0 | -13.64 | 0.17 | 0 | yes | yes | **p2_efficientnet_b0 better** |
| b0_simple_cnn | p3_convnext_tiny | -17.40 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 3/1) |
| p1_resnet50 | p2_efficientnet_b0 | +1.23 | 0.11 | 8.95e-08 | yes | yes | **p1_resnet50 better** |
| p1_resnet50 | p3_convnext_tiny | -2.53 | n/a | 1.66e-37 | no | yes | preliminary: needs >= 2 seeds per arm (has 3/1) |
| p2_efficientnet_b0 | p3_convnext_tiny | -3.76 | n/a | 3.34e-71 | no | yes | preliminary: needs >= 2 seeds per arm (has 3/1) |
| e1_resnet50_linear_probe | p1_resnet50 | -15.48 | 0.11 | 0 | yes | yes | **p1_resnet50 better** |
| e2_resnet50_label_smoothing | p1_resnet50 | +0.39 | 0.18 | 0.0187 | yes | yes | **e2_resnet50_label_smoothing better** |
| e3_b0_simple_cnn_frac10 | b0_simple_cnn | -46.19 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |
| e3_b0_simple_cnn_frac25 | b0_simple_cnn | -23.55 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |
| e3_b0_simple_cnn_frac50 | b0_simple_cnn | -9.81 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |
| e3_p1_resnet50_frac10 | p1_resnet50 | -17.96 | n/a | 0 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |
| e3_p1_resnet50_frac25 | p1_resnet50 | -8.58 | n/a | 3.29e-282 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |
| e3_p1_resnet50_frac50 | p1_resnet50 | -3.24 | n/a | 1.08e-58 | no | yes | preliminary: needs >= 2 seeds per arm (has 1/3) |

## Pre-registered hypotheses

- **E1 e1_resnet50_linear_probe** (vs p1_resnet50; changed: backbone frozen (BN in eval mode), only the new fc is trained): Full fine-tuning beats a frozen ImageNet backbone by >= 5 pp test macro-F1, because ImageNet features under-represent fine food texture.
- **E2 e2_resnet50_label_smoothing** (vs p1_resnet50; changed: train.label_smoothing 0.0 -> 0.1): Label smoothing improves test macro-F1 by >= 0.5 pp and lowers ECE, because train labels are noisy while test labels are clean.
- **E3 e3_b0_simple_cnn_frac10** (vs b0_simple_cnn; changed: data.train_fraction 1.0 -> 0.1 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
- **E3 e3_b0_simple_cnn_frac25** (vs b0_simple_cnn; changed: data.train_fraction 1.0 -> 0.25 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
- **E3 e3_b0_simple_cnn_frac50** (vs b0_simple_cnn; changed: data.train_fraction 1.0 -> 0.5 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
- **E3 e3_p1_resnet50_frac10** (vs p1_resnet50; changed: data.train_fraction 1.0 -> 0.1 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
- **E3 e3_p1_resnet50_frac25** (vs p1_resnet50; changed: data.train_fraction 1.0 -> 0.25 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
- **E3 e3_p1_resnet50_frac50** (vs p1_resnet50; changed: data.train_fraction 1.0 -> 0.5 (stratified per class, subset_seed 42)): Pretrained models are far more data-efficient: with 10 % of the train data P1 keeps >= 85 % of its full-data macro-F1, B0 does not.
