"""Spatiotemporal tube masking for 3D VideoMAE-style pretraining."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


class TubeMaskingGenerator3D:
    """
    Tube masking on a 3D token grid (D, H, W).

    For each fixed (H, W) location, all temporal indices D share the same
    mask state (visible vs masked), matching VideoMAE tube masking.

    Args:
        token_shape: (D, H, W) number of tokens along each axis.
        mask_ratio: fraction of spatial tubes to mask (default 0.95 -> 5% visible).
        rng: optional numpy Generator for reproducibility.
    """

    def __init__(
        self,
        token_shape: Union[Sequence[int], Tuple[int, int, int]],
        mask_ratio: float = 0.95,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.token_shape = tuple(int(x) for x in token_shape)
        if len(self.token_shape) != 3:
            raise ValueError(f"token_shape must be (D, H, W), got {self.token_shape}")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        self.mask_ratio = mask_ratio
        self.rng = rng or np.random.default_rng()

        d, h, w = self.token_shape
        self.num_tubes = h * w
        self.num_tokens = d * h * w
        self.num_masked_tubes = int(math.floor(self.num_tubes * mask_ratio))
        self.num_masked_tubes = min(max(self.num_masked_tubes, 0), self.num_tubes - 1)
        self.num_visible_tubes = self.num_tubes - self.num_masked_tubes

    def __repr__(self) -> str:
        return (
            f"TubeMaskingGenerator3D(shape={self.token_shape}, mask_ratio={self.mask_ratio}, "
            f"masked_tubes={self.num_masked_tubes}/{self.num_tubes})"
        )

    def _tube_mask_hw(self) -> np.ndarray:
        """Boolean [H, W]: True = masked tube (to reconstruct), False = visible."""
        h, w = self.token_shape[1], self.token_shape[2]
        mask_hw = np.zeros((h, w), dtype=bool)
        flat = mask_hw.reshape(-1)
        perm = self.rng.permutation(self.num_tubes)
        flat[perm[: self.num_masked_tubes]] = True
        return mask_hw

    def __call__(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns bool mask [B, 1, D, H, W] where True = masked (target for reconstruction).
        """
        d, h, w = self.token_shape
        masks = []
        for _ in range(batch_size):
            tube_hw = self._tube_mask_hw()
            tube_dhw = np.broadcast_to(tube_hw[np.newaxis, :, :], (d, h, w))
            masks.append(tube_dhw)
        arr = np.stack(masks, axis=0)[:, np.newaxis, ...]
        out = torch.from_numpy(arr)
        if device is not None:
            out = out.to(device)
        return out

    def visible_token_count(self, batch_size: int = 1) -> int:
        return batch_size * self.num_visible_tubes * self.token_shape[0]

    @staticmethod
    def token_shape_from_video(
        spatial_size: Union[Sequence[int], Tuple[int, int, int]],
        patch_size: int = 2,
        num_downsample_stages: int = 4,
    ) -> Tuple[int, int, int]:
        """
        Token grid at Swin-UNETR bottleneck (patch grid / 2^num_downsample_stages).
        Default matches img (96,128,128), patch=2, 4 merges -> (3,4,4).
        """
        d, h, w = (int(spatial_size[0]), int(spatial_size[1]), int(spatial_size[2]))
        factor = patch_size * (2**num_downsample_stages)
        return (d // factor, h // factor, w // factor)

    @staticmethod
    def patch_token_shape(
        spatial_size: Union[Sequence[int], Tuple[int, int, int]],
        patch_size: int = 2,
    ) -> Tuple[int, int, int]:
        d, h, w = (int(spatial_size[0]), int(spatial_size[1]), int(spatial_size[2]))
        return (d // patch_size, h // patch_size, w // patch_size)
