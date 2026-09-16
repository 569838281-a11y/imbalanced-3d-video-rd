"""Unsharp masking (UM) for 3D ophthalmic ultrasound video tensors."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def apply_unsharp_masking(
    video: torch.Tensor,
    strength: float = 1.5,
    blur_ksize: tuple[int, int] = (5, 5),
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Unsharp mask on [C, D, H, W] video: Sharpened = (1 + s) * Original - s * Blurred.

    Each frame (D axis) is blurred with OpenCV GaussianBlur on H x W, per channel.
    Output is clamped to [0, 1] if input max <= 1.5, else [0, 255].
    """
    if video.dim() != 4:
        raise ValueError(f"Expected video shape [C, D, H, W], got {tuple(video.shape)}")

    x = video.detach().float().cpu().numpy()
    use_255 = float(x.max()) > 1.5
    hi = 255.0 if use_255 else 1.0

    c, d, _, _ = x.shape
    out = np.empty_like(x, dtype=np.float32)
    ksize = (int(blur_ksize[0]), int(blur_ksize[1]))
    if ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
        ksize = (ksize[0] | 1, ksize[1] | 1)

    for di in range(d):
        for ci in range(c):
            frame = x[ci, di].astype(np.float32, copy=False)
            blurred = cv2.GaussianBlur(frame, ksize, blur_sigma)
            sharpened = (1.0 + strength) * frame - strength * blurred
            out[ci, di] = np.clip(sharpened, 0.0, hi)

    return torch.from_numpy(out).to(device=video.device, dtype=video.dtype)
