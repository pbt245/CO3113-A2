# CO3133 Assignment 2 - Food-101 Image Classification

Deep Learning on Large-Scale Data and Specialized Tasks - Group G-roundhog, Semester 261.

**Current milestone: M1 - Dataset Proposal (awaiting instructor approval).**
Implementation starts after approval.

- Dataset proposal: [PROPOSAL.md](PROPOSAL.md)
- Assignment page: <https://pbt245.github.io/CO3133-DLIA/assignments/assignment2.html>
- Assignment specification: [instruction.md](instruction.md)

## Summary of the proposal

| Item | Choice |
| --- | --- |
| Track | Image classification |
| Dataset | Food-101 (101 classes, 101,000 images) |
| Split | 68,175 train / 7,575 val / 25,250 official test; stratified, split seed 42, split unit = image |
| Metrics | Top-1 accuracy, macro-F1 (primary), top-5, per-class F1, compute cost |
| Baseline | SimpleCNN from scratch (≈ 1.2 M params) |
| Pretrained | ResNet-50 and EfficientNet-B0 (ImageNet-1k), full fine-tuning |
| Mandatory ablation | Frozen backbone (linear probe) vs full fine-tuning |
| Compute | Kaggle T4 16 GB; ≈ 8.5 GPU-h mandatory, ≈ 14 GPU-h with optional runs |

## Members

| Name | Student ID | GitHub | Email |
| --- | --- | --- | --- |
| Phan Bá Thanh | 2353084 | [pbt245](https://github.com/pbt245) | thanh.phancsbk@hcmut.edu.vn |
| Trần Công Hoàng Phước | 2352966 | [BenjaminPhuoc](https://github.com/BenjaminPhuoc) | phuoc.tranbenjamin@hcmut.edu.vn
