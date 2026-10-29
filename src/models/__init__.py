"""Model registry: B0 (from scratch) and the pretrained backbones, built from a config section."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.models.pretrained import BACKBONES, PretrainedClassifier
from src.models.simple_cnn import SimpleCNN

MODEL_NAMES = ("simple_cnn", *BACKBONES)


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    cfg = copy.deepcopy(model_cfg)
    name = str(cfg.pop("name")).lower()
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **cfg)
    if name in BACKBONES:
        return PretrainedClassifier(name, num_classes=num_classes, **cfg)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> dict:
    return {
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total": int(sum(p.numel() for p in model.parameters())),
    }


def count_macs(model: nn.Module, image_size: int = 224) -> float:
    """Multiply-accumulates per image (the number torchvision reports as "GFLOPS")."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    model.train(was_training)
    return fc.get_total_flops() / 2
