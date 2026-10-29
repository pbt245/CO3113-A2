"""B0 - self-designed CNN baseline trained from scratch (no pretrained weights).

Block k:  Conv3x3(pad=1) -> BN -> ReLU -> Conv3x3(pad=1) -> BN -> ReLU -> MaxPool2x2 -> Dropout2d
  * 3x3 convolutions keep H, W (padding=1) and learn local filters shared over all positions
  * max pooling halves H and W, giving some translation tolerance and a larger receptive field
Head: global average pooling -> Dropout -> Linear -> logits
Food-101 with channels [32, 64, 128, 256] at 224 px: 224 -> 112 -> 56 -> 28 -> 14 -> GAP.
"""
from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        super().__init__(*layers)


class SimpleCNN(nn.Module):
    head_prefix = "head"

    def __init__(self, num_classes: int, in_channels: int = 3, channels=(32, 64, 128, 256),
                 block_dropout: float = 0.0, head_dropout: float = 0.3):
        super().__init__()
        blocks = []
        c = in_channels
        for out in channels:
            blocks.append(ConvBlock(c, int(out), block_dropout))
            c = int(out)
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(float(head_dropout)), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
