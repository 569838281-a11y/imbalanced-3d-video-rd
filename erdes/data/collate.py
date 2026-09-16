"""Batch collation for variable-resolution video tensors (before GPU resize)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_video_batch_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad clips in a batch to common (D, H, W) so default stack works.
    CPU only pads; trilinear resize to target size runs on GPU.
    """
    videos, labels = zip(*batch)
    max_d = max(int(v.shape[1]) for v in videos)
    max_h = max(int(v.shape[2]) for v in videos)
    max_w = max(int(v.shape[3]) for v in videos)

    padded: list[torch.Tensor] = []
    for v in videos:
        c, d, h, w = v.shape
        pad = (0, max_w - w, 0, max_h - h, 0, max_d - d)
        if any(pad):
            v = F.pad(v, pad)
        padded.append(v)

    return torch.stack(padded, dim=0), torch.stack(labels, dim=0)
