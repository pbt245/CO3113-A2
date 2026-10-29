"""ImageNet-pretrained torchvision backbones with a new 101-way linear head.

Fine-tuning modes
  * full fine-tuning  : every parameter is trained; the new head and the backbone get separate
                        learning rates (discriminative LR, optim.lr vs optim.backbone_lr)
  * frozen backbone   : `freeze_backbone: true` -> only the head is trained (linear probe, E1).
                        The backbone stays in eval mode, so BatchNorm running statistics are not
                        updated by Food-101 batches either.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

# name -> (constructor, weights enum member, attribute path of the final Linear layer)
BACKBONES = {
    "resnet50": (tvm.resnet50, "ResNet50_Weights.IMAGENET1K_V2", "fc"),
    "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights.IMAGENET1K_V1", "classifier.1"),
    "convnext_tiny": (tvm.convnext_tiny, "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "classifier.2"),
}


def _weights(spec: str):
    enum_name, member = spec.split(".")
    return getattr(getattr(tvm, enum_name), member)


class PretrainedClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False,
                 head_dropout: float = 0.0):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"Unknown backbone '{arch}'. Choose from {list(BACKBONES)}")
        ctor, weights, head_path = BACKBONES[arch]
        self.net = ctor(weights=_weights(weights) if pretrained else None)
        self.weights = weights if pretrained else None
        parent_path, _, attr = head_path.rpartition(".")
        parent = self.net.get_submodule(parent_path) if parent_path else self.net
        old = getattr(parent, attr)
        new_head = nn.Linear(old.in_features, num_classes)
        setattr(parent, attr, nn.Sequential(nn.Dropout(head_dropout), new_head) if head_dropout else new_head)
        self.head_path = head_path
        self.head_prefix = f"net.{head_path}"
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for name, p in self.named_parameters():
                p.requires_grad = name.startswith(self.head_prefix)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.net.eval()
            self.net.get_submodule(self.head_path).train()
        return self

    def forward(self, x):
        return self.net(x)
