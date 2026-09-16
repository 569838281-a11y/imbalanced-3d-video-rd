"""GPU-side 3D video pad + resize + normalize (after DataLoader transfer)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _pad_to_square_3d(x: torch.Tensor) -> torch.Tensor:
    """x: [C, D, H, W] — symmetric pad so H == W."""
    _, _, h, w = x.shape
    if h == w:
        return x
    diff = abs(h - w)
    pad_left = diff // 2
    pad_right = diff - pad_left
    if h < w:
        return F.pad(x, (0, 0, pad_left, pad_right))
    return F.pad(x, (pad_left, pad_right, 0, 0))


def gpu_resize_video(x: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Resize one clip on GPU.
    x: [C, D, H, W] (any D before resample; D dimension is temporal depth).
    size: target (D, H, W).
    """
    d, h, w = size
    x = _pad_to_square_3d(x)
    return F.interpolate(x.unsqueeze(0), size=(d, h, w), mode="trilinear", align_corners=False).squeeze(0)


def gpu_preprocess_batch(
    videos: torch.Tensor,
    target_size: Tuple[int, int, int],
    normalize: bool = True,
) -> torch.Tensor:
    """
    videos: [B, C, D, H, W] on GPU; may be at decode resolution.
    Returns float tensor at target_size (normalized to [0,1] if normalize).
    """
    if videos.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got shape {tuple(videos.shape)}")

    td, th, tw = target_size
    if videos.shape[2:] == (td, th, tw):
        out = videos.float()
    else:
        out = torch.stack([gpu_resize_video(videos[i], target_size) for i in range(videos.size(0))])

    if normalize:
        if out.max() > 1.5:
            out = out / 255.0
    return out
