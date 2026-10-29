# EDA summary - Food-101

101 classes, 101,000 images. Split seed 42, fingerprints {'train': '8f130b6f7c76d577', 'val': 'bdded6761644591e', 'test': '101fc179791da757'}.

## Class balance

| Split | Images | Min / class | Max / class | Max/min |
|---|---|---|---|---|
| train | 68,189 | 675 | 676 | 1.001 |
| val | 7,561 | 74 | 75 | 1.014 |
| test | 25,250 | 250 | 250 | 1.000 |

## Image geometry (raw archive)

| Quantity | Mean | Min | p5 | Median | p95 | Max |
|---|---|---|---|---|---|---|
| width (px) | 495.8 | 193 | 384.00 | 512.00 | 512.00 | 512.00 |
| height (px) | 475.4 | 122 | 341.00 | 512.00 | 512.00 | 512.00 |
| aspect w/h | 1.1 | 0.38 | 0.75 | 1.00 | 1.50 | 4.20 |
| file size (KB) | 49.5 | 8 | 28.70 | 48.42 | 73.57 | 210.65 |

- max side = 512 px: 100.0 % of images
- shorter side < 224 px: 0.01 %
- landscape / square / portrait: 25.4 / 62.9 / 11.7 %
- raw colour modes: {'RGB': 100997, 'L': 3}

## Pixel statistics

| Split | Mean RGB | Std RGB | Brightness | Saturation | Low-sat. (<0.05) |
|---|---|---|---|---|---|
| train | 0.540 / 0.442 / 0.344 | 0.270 / 0.272 / 0.277 | 0.463 | 0.231 | 0.51 % |
| val | 0.538 / 0.440 / 0.342 | 0.272 / 0.273 / 0.276 | 0.460 | 0.231 | 0.54 % |
| test | 0.543 / 0.445 / 0.346 | 0.266 / 0.269 / 0.276 | 0.466 | 0.230 | 0.43 % |

Darkest classes: {'bibimbap': 0.387, 'miso_soup': 0.399, 'mussels': 0.4, 'prime_rib': 0.416, 'edamame': 0.417}

Brightest classes: {'omelette': 0.5, 'eggs_benedict': 0.501, 'spaghetti_bolognese': 0.503, 'risotto': 0.513, 'spaghetti_carbonara': 0.537}

## Near-duplicates (pHash, Hamming <= 4)

| Pair type | Pairs |
|---|---|
| test-test | 6 |
| test-train | 39 |
| test-val | 9 |
| train-train | 88 |
| val-val | 1 |

- pairs whose two images carry different labels: 4
- val images moved to train by make_splits: 14
- test images flagged (near-duplicate of train/val): 47 (0.19 % of test)
