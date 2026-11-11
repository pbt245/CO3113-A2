"""Grad-CAM (Selvaraju et al., ICCV 2017) for the B0 / pretrained classifiers.

CAM_c = ReLU( sum_k  mean_{h,w}(d y_c / d A_k) * A_k ),  A = feature maps of the last convolutional stage,
upsampled to the input size and scaled to [0, 1]. Self-implemented with forward / backward hooks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def target_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature stage (output N x C x 7 x 7 for the pretrained models at 224 px, 14 x 14 for B0)."""
    if hasattr(model, "net"):
        net = model.net
        return net.layer4 if hasattr(net, "layer4") else net.features
    return model.features


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module | None = None):
        self.model = model.eval()
        self.layer = layer if layer is not None else target_layer(model)
        self._acts = self._grads = None
        self._h = [self.layer.register_forward_hook(self._save_act)]

    def _save_act(self, module, inp, out):
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 3, H, W) normalized input. Returns (cam HxW in [0, 1], class index, softmax prob)."""
        x = x.requires_grad_(False)
        with torch.enable_grad():
            logits = self.model(x)
            probs = logits.softmax(dim=1)
            c = int(logits.argmax(dim=1)) if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, c].backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), c, float(probs[0, c].detach())

    def remove(self):
        for h in self._h:
            h.remove()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """img: (H, W, 3) uint8; cam: (H, W) in [0, 1] -> (H, W, 3) uint8 with a jet heat map."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (255 * ((1 - alpha) * img / 255.0 + alpha * heat)).clip(0, 255).astype(np.uint8)
