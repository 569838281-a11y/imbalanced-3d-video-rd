"""3D Grad-CAM for ResNet3D (MONAI encoder + topk/avg head)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_target_layer(net: nn.Module) -> nn.Module:
    """Last 3D conv block before the classification head."""
    if hasattr(net, "enc") and hasattr(net.enc, "layer4"):
        return net.enc.layer4
    if hasattr(net, "model") and hasattr(net.model, "layer4"):
        return net.model.layer4
    raise AttributeError("Could not find ResNet3D layer4 for Grad-CAM.")


class GradCAM3D:
    def __init__(self, net: nn.Module) -> None:
        self.net = net
        self.target_layer = _resolve_target_layer(net)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _forward_hook(self, _module, _inp, output) -> None:
        self._activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def __call__(self, video: torch.Tensor, label: int) -> np.ndarray:
        """
        video: [1, C, D, H, W] on same device as net.
        label: 0 (non-RD) or 1 (RD) — backward uses the true-class score.
        Returns CAM volume [D, H, W] normalized to [0, 1].
        """
        self.net.eval()
        self._activations = None
        self._gradients = None

        video = video.detach().requires_grad_(True)
        logits = self.net(video)
        score = logits.view(-1)[0]
        if int(label) == 0:
            score = -score

        self.net.zero_grad(set_to_none=True)
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        grads = self._gradients
        acts = self._activations
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=video.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-8)
        return cam_np


def overlay_cam_on_frame(
    gray_frame: np.ndarray,
    cam_frame: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Return RGB uint8 overlay image."""
    gray = np.clip(gray_frame, 0.0, 1.0)
    cam = np.clip(cam_frame, 0.0, 1.0)
    base = np.stack([gray, gray, gray], axis=-1)
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]
    out = (1.0 - alpha) * base + alpha * heat
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def save_heatmap_png(
    video: torch.Tensor,
    cam: np.ndarray,
    out_path: Path,
    frame_index: Optional[int] = None,
    alpha: float = 0.45,
) -> None:
    """
    video: [C, D, H, W] float tensor (normalized 0-1).
    cam: [D, H, W] in [0, 1].
    """
    if video.ndim != 4:
        raise ValueError(f"Expected video [C,D,H,W], got {tuple(video.shape)}")

    d = video.shape[1]
    idx = d // 2 if frame_index is None else int(frame_index)
    idx = max(0, min(idx, d - 1))

    gray = video[0, idx].detach().cpu().numpy()
    heat = cam[idx]
    rgb = overlay_cam_on_frame(gray, heat, alpha=alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
