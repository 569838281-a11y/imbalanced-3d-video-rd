"""Load preprocessed video tensors from D: cache (fast path)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from .erdes_dataset import VideoDataset
from .unsharp_masking import apply_unsharp_masking


def cache_key_for_path(relative_path: str) -> str:
    return hashlib.md5(relative_path.replace("\\", "/").encode()).hexdigest()


def cache_file_path(cache_root: Path, relative_path: str) -> Path:
    return cache_root / f"{cache_key_for_path(relative_path)}.pt"


class CachedVideoDataset(Dataset):
    """
    Same labels/paths as VideoDataset; reads tensors from disk cache when present.
  Falls back to on-the-fly decode for missing entries.
    """

    def __init__(
        self,
        csv_path: str,
        size: Union[int, Tuple[int, int, int]],
        data_root: str = "",
        cache_dir: str = "",
        video_column: str = "path",
        label_column: str = "label",
        strict_cache: bool = False,
        cache_mmap: bool = True,
        use_um: bool = False,
        um_strength: float = 1.5,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.video_paths = self.df[video_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.size = tuple(int(x) for x in size)
        self.data_root = data_root
        self.cache_root = Path(cache_dir) if cache_dir else None
        self.strict_cache = strict_cache
        self.cache_mmap = cache_mmap
        self.use_um = use_um
        self.um_strength = um_strength
        self._fallback = VideoDataset(
            csv_path=csv_path,
            size=self.size,
            data_root=data_root,
            video_column=video_column,
            label_column=label_column,
            use_um=use_um,
            um_strength=um_strength,
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_root is None:
            return None
        rel = self.video_paths[idx].replace("\\", "/")
        return cache_file_path(self.cache_root, rel)

    def __getitem__(self, idx: int):
        cache_path = self._cache_path(idx)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if cache_path is not None and cache_path.is_file():
            mmap = self.cache_mmap
            obj = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
                **({"mmap": True} if mmap else {}),
            )
            video = obj["video"] if isinstance(obj, dict) else obj
            if self.use_um:
                video = apply_unsharp_masking(video.float(), strength=self.um_strength)
            return video, label

        if self.strict_cache:
            raise FileNotFoundError(f"Cache miss (strict): {cache_path}")

        return self._fallback[idx]
