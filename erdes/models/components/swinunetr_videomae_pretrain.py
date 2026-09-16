"""VideoMAE-style self-supervised pretraining with Swin-UNETR encoder."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.swinunetr import SwinUnetrEncoder
from .videomae_mask import TubeMaskingGenerator3D


class TransformerBlock3D(nn.Module):
    """Standard pre-norm Transformer block on token sequences."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder3D(nn.Module):
    """
    Lightweight MAE decoder: 3 Transformer blocks + per-token pixel prediction.
    Reconstructs full video from bottleneck tokens + [MASK] placeholders.
    """

    def __init__(
        self,
        encoder_dim: int,
        token_shape: Tuple[int, int, int],
        patch_volume: Tuple[int, int, int],
        in_channels: int = 1,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.token_shape = token_shape
        self.patch_volume = patch_volume
        self.in_channels = in_channels
        d, h, w = token_shape
        self.num_tokens = d * h * w
        self.patch_pixels = (
            int(patch_volume[0]) * int(patch_volume[1]) * int(patch_volume[2]) * in_channels
        )

        self.encoder_proj = nn.Linear(encoder_dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock3D(encoder_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(encoder_dim)
        self.pred = nn.Linear(encoder_dim, self.patch_pixels)

    def forward(
        self,
        enc_tokens: torch.Tensor,
        tube_mask: torch.Tensor,
        patches_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_tokens: [B, C, D, H, W] encoder bottleneck features.
            tube_mask: [B, 1, D, H, W] bool, True = masked (needs reconstruction).

        Returns:
            pred_video: [B, in_channels, D_vol, H_vol, W_vol]
            tube_mask: same mask for loss computation
        """
        b, c, d, h, w = enc_tokens.shape
        x = enc_tokens.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
        x = self.encoder_proj(x)

        mask_flat = tube_mask.reshape(b, -1)
        mask_token = self.mask_token.expand(b, x.shape[1], -1)
        x = torch.where(mask_flat.unsqueeze(-1), mask_token, x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patch_pred = self.pred(x)

        if patches_only:
            return patch_pred, tube_mask

        pd, ph, pw = self.patch_volume
        pred = patch_pred.view(b, d, h, w, self.in_channels, pd, ph, pw)
        pred = pred.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        pred = pred.view(b, self.in_channels, d * pd, h * ph, w * pw)
        return pred, tube_mask


class SwinUNETRVideoMAEPretrain(nn.Module):
    """
    VideoMAE pretraining: tube-masked input -> Swin-UNETR encoder (visible tubes only)
    -> MAE decoder -> MSE on masked tube pixels.
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int] = (96, 128, 128),
        in_channels: int = 1,
        feature_size: int = 24,
        mask_ratio: float = 0.95,
        patch_size: int = 2,
        num_downsample_stages: int = 4,
        decoder_heads: int = 8,
        decoder_layers: int = 3,
        use_checkpoint: bool = False,
        patch_level_loss: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = tuple(int(x) for x in img_size)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_level_loss = patch_level_loss
        self.encoder = SwinUnetrEncoder(
            img_size=self.img_size,
            in_channels=in_channels,
            feature_size=feature_size,
            norm_name="instance",
            normalize=True,
            use_checkpoint=use_checkpoint,
        )
        enc_dim = self.encoder.bottle_neck_embed_dim
        self.token_shape = TubeMaskingGenerator3D.token_shape_from_video(
            self.img_size, patch_size=patch_size, num_downsample_stages=num_downsample_stages
        )
        pd = self.img_size[0] // self.token_shape[0]
        ph = self.img_size[1] // self.token_shape[1]
        pw = self.img_size[2] // self.token_shape[2]
        self.patch_volume = (pd, ph, pw)

        self.mask_generator = TubeMaskingGenerator3D(self.token_shape, mask_ratio=mask_ratio)
        self.decoder = MAEDecoder3D(
            encoder_dim=enc_dim,
            token_shape=self.token_shape,
            patch_volume=self.patch_volume,
            in_channels=in_channels,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
        )

    def video_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,D,H,W] -> [B, num_tokens, patch_pixels]."""
        b, c, d0, h0, w0 = x.shape
        pd, ph, pw = self.patch_volume
        d, h, w = self.token_shape
        if d0 != d * pd or h0 != h * ph or w0 != w * pw:
            x = F.interpolate(x, size=(d * pd, h * ph, w * pw), mode="trilinear", align_corners=False)
        patches = x.view(b, c, d, pd, h, ph, w, pw)
        return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, d * h * w, c * pd * ph * pw)

    def tube_mask_to_pixel_mask(self, tube_mask: torch.Tensor, spatial: Tuple[int, int, int]) -> torch.Tensor:
        """Expand bottleneck tube mask [B,1,D_t,H_t,W_t] to pixel [B,1,D,H,W]."""
        b = tube_mask.shape[0]
        d, h, w = spatial
        m = tube_mask.float()
        if m.shape[2:] != (d, h, w):
            m = F.interpolate(m, size=(d, h, w), mode="nearest")
        return m > 0.5

    def forward_encoder_visible_only(
        self,
        x: torch.Tensor,
        tube_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encoder path: zero-out masked tubes in pixel space (tube-consistent along D),
        then run Swin-UNETR on visible content only (masked regions carry no signal).
        """
        pixel_mask = self.tube_mask_to_pixel_mask(tube_mask, x.shape[2:])
        x_visible = x * (~pixel_mask).to(dtype=x.dtype)
        return self.encoder(x_visible)

    def forward(
        self,
        x: torch.Tensor,
        tube_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, D, H, W] ultrasound video.

        Returns:
            loss: scalar MSE on masked tube pixels.
            pred: reconstructed video.
            tube_mask: [B, 1, D_t, H_t, W_t] bool mask used.
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_channels:
            if x.shape[1] > 1 and self.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            else:
                raise ValueError(f"Expected in_channels={self.in_channels}, got {x.shape[1]}")

        b = x.shape[0]
        if tube_mask is None:
            tube_mask = self.mask_generator(b, device=x.device)

        enc_feat = self.forward_encoder_visible_only(x, tube_mask)

        if self.patch_level_loss:
            patch_pred, tube_mask = self.decoder(enc_feat, tube_mask, patches_only=True)
            target_patches = self.video_to_patches(x)
            mask_flat = tube_mask.reshape(x.shape[0], -1)
            if mask_flat.any():
                loss = F.mse_loss(patch_pred[mask_flat], target_patches[mask_flat])
            else:
                loss = F.mse_loss(patch_pred, target_patches)
            pred = patch_pred
        else:
            pred, tube_mask = self.decoder(enc_feat, tube_mask)
            pixel_mask = self.tube_mask_to_pixel_mask(tube_mask, x.shape[2:])
            target = x
            if pred.shape != target.shape:
                pred = F.interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
            recon_mask = pixel_mask.expand_as(target)
            diff = (pred - target) ** 2
            loss = diff[recon_mask].mean() if recon_mask.any() else diff.mean()

        return loss, pred, tube_mask

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        return self.encoder(x)
