# Comparison - food101

Mean ± sample std over training seeds (same split for every run). Checkpoint = best validation macro-F1; the test set is evaluated once per checkpoint. `*` = optional experiment. "w/o dups" excludes the test images flagged as pHash near-duplicates of train/val. Hardware: ['NVIDIA GeForce RTX 3060']; inference timing fp16 on CUDA = [True].

## Main models

| ID | Experiment | Seeds | Params (M) | GMACs | Test acc (%) | Test macro-F1 (%) | Top-5 (%) | ECE | Macro-F1 w/o dups (%) | Best epoch | Train min/epoch | Train img/s | Infer img/s (bs128) | Latency bs1 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | b0_simple_cnn | 42 | 1.20 | 2.59 | 71.35 | 71.26 | 91.40 | 0.081 | 71.25 | 30.0 | 2.6 | 442 | 1769 | 0.77 |
| P1 | p1_resnet50 | 42 | 23.71 | 4.09 | 86.36 | 86.38 | 97.16 | 0.047 | 86.37 | 10.0 | 3.8 | 301 | 909 | 2.71 |

## Traceability

Commits used: ['f4968df8904b29607d07b73e04e3871c45b887c6']
Runs with uncommitted changes: 2
Train split fingerprints: ['8f130b6f7c76d577']
