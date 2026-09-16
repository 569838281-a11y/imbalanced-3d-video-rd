"""MONAI Swin-UNETR with official SSL pretrained encoder + classification head."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .cls_model import ClassificationHead

logger = logging.getLogger(__name__)

MONAI_SSL_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/"
    "ssl_pretrained_weights.pth"
)
DEFAULT_SSL_WEIGHTS_PATH = Path("weights/ssl_pretrained_weights.pth")

# Official SSL checkpoint uses feature_size=48 (see MONAI SwinUNETR docs).
MONAI_SSL_FEATURE_SIZE = 48


def download_monai_ssl_weights(
    dest: Path = DEFAULT_SSL_WEIGHTS_PATH,
    url: str = MONAI_SSL_WEIGHTS_URL,
) -> Path:
    """Download SSL weights if missing; return local path."""
    dest = Path(dest)
    if dest.is_file():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None

    try:
        from monai.apps import download_url

        download_url(url, str(dest))
        if dest.is_file():
            return dest
    except Exception as exc:
        last_err = exc
        logger.warning("monai.apps.download_url failed: %s", exc)

    try:
        torch.hub.download_url_to_file(url, str(dest), progress=True)
        if dest.is_file():
            return dest
    except Exception as exc:
        last_err = exc
        raise RuntimeError(
            f"Failed to download MONAI SSL weights to {dest}. "
            f"Download manually from:\n  {url}\n"
            f"Last error: {last_err}"
        ) from exc

    return dest


def load_monai_ssl_checkpoint(
    weights_path: Optional[Union[str, Path]] = None,
    map_location: str = "cpu",
) -> dict:
    """Load raw MONAI SSL .pth checkpoint."""
    path = Path(weights_path) if weights_path else download_monai_ssl_weights()
    if not path.is_file():
        path = download_monai_ssl_weights(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected SSL checkpoint format at {path}")
    return ckpt


def apply_monai_ssl_weights_to_swinunetr(
    model: SwinUNETR,
    weights_path: Optional[Union[str, Path]] = None,
) -> tuple[int, int]:
    """
    Inject SSL weights into SwinUNETR encoder (swinViT).

    Supports both legacy ``load_from`` checkpoints (state_dict + module.* keys)
    and newer releases (model dict with encoder.* keys via copy_model_state).
    """
    ckpt = load_monai_ssl_checkpoint(weights_path)

    # Legacy format used by SwinUNETR.load_from()
    if "state_dict" in ckpt and any(
        k.startswith("module.") for k in ckpt["state_dict"]
    ):
        model.load_from(ckpt)
        return -1, -1

    from monai.networks.nets.swin_unetr import filter_swinunetr
    from monai.networks.utils import copy_model_state

    src = ckpt.get("model", ckpt.get("state_dict", ckpt))
    _dst, loaded, not_loaded = copy_model_state(model, src, filter_func=filter_swinunetr)
    return len(loaded), len(not_loaded)


class MonaiSSLSwinUNETRClassifier(nn.Module):
    """
    Binary/multi-class classifier built on MONAI Swin-UNETR encoder (swinViT) with SSL weights.
    Input: [B, 1, D, H, W] — project default (96, 128, 128).
    """

    def __init__(
        self,
        num_classes: int = 1,
        feature_size: int = MONAI_SSL_FEATURE_SIZE,
        pooling: str = "avg",
        topk_ratio: float = 0.5,
        ssl_weights_path: Optional[Union[str, Path]] = None,
        load_ssl: bool = True,
        use_checkpoint: bool = False,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.backbone = SwinUNETR(
            in_channels=1,
            out_channels=out_channels,
            feature_size=feature_size,
            norm_name="instance",
            normalize=True,
            use_checkpoint=use_checkpoint,
        )

        if load_ssl:
            n_loaded, n_miss = apply_monai_ssl_weights_to_swinunetr(
                self.backbone, ssl_weights_path
            )
            if n_loaded >= 0:
                logger.info(
                    "Loaded MONAI SSL weights via copy_model_state: %d tensors (%d skipped).",
                    n_loaded,
                    n_miss,
                )
            else:
                logger.info("Loaded MONAI SSL weights via SwinUNETR.load_from().")

        self.feature_dim = 16 * feature_size
        self.cls = ClassificationHead(
            input_dim=self.feature_dim,
            hidden_size=self.feature_dim // 2,
            num_classes=num_classes,
            pooling=pooling,
            topk_ratio=topk_ratio,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone.swinViT(x, self.backbone.normalize)
        return hidden_states[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.encode(x))


def build_monai_ssl_swinunetr_classifier(
    num_classes: int = 1,
    pooling: str = "avg",
    topk_ratio: float = 0.5,
    load_ssl: bool = True,
    ssl_weights_path: Optional[str] = None,
    feature_size: int = MONAI_SSL_FEATURE_SIZE,
    use_checkpoint: bool = False,
) -> MonaiSSLSwinUNETRClassifier:
    """Hydra-compatible factory."""
    return MonaiSSLSwinUNETRClassifier(
        num_classes=num_classes,
        feature_size=feature_size,
        pooling=pooling,
        topk_ratio=topk_ratio,
        load_ssl=load_ssl,
        ssl_weights_path=ssl_weights_path,
        use_checkpoint=use_checkpoint,
    )
