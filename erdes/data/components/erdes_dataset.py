import os
from typing import Tuple, Union

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .unsharp_masking import apply_unsharp_masking
from .utils import resize


def _read_video_frames(
    video_path: str,
    num_frames: int = 96,
    max_decode: int = 160,
) -> torch.Tensor:
    """Uniformly sample num_frames; cap raw decode to avoid RAM spikes on long clips."""
    decoded: list[np.ndarray] = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_ndarray(format="rgb24"))
            if len(decoded) >= max_decode:
                break

    if not decoded:
        raise ValueError(f"No frames decoded from video: {video_path}")

    if len(decoded) <= num_frames:
        indices = np.arange(len(decoded))
    else:
        indices = np.linspace(0, len(decoded) - 1, num_frames, dtype=int)

    sampled = np.stack([decoded[i] for i in indices])
    return torch.from_numpy(sampled)


class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        size: Union[int, Tuple[int, int, int]],  # Desired (D, H, W)
        data_root: str = "",
        video_column: str = "path",
        label_column: str = "label",
        defer_resize: bool = False,
        use_um: bool = False,
        um_strength: float = 1.5,
    ):
        self.df = pd.read_csv(csv_path)
        self.video_paths = self.df[video_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.size = tuple(int(x) for x in size)
        self.defer_resize = defer_resize
        self.use_um = use_um
        self.um_strength = um_strength
        self.resize_tf = None if defer_resize else resize(self.size)
        self.data_root = data_root

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        if self.data_root:
            video_path = os.path.join(self.data_root, video_path)
        label = self.labels[idx]

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Read video: returns (T, H, W, C)
        depth = self.size[0]
        video = _read_video_frames(video_path, num_frames=depth, max_decode=max(depth * 2, 160))

        # Convert to float tensor and permute to [C, D, H, W]
        video = video.float()  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)  # [C, D, H, W]

        # If video has 3 channels, convert to grayscale by averaging
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)  # [1, D, H, W]

        if self.resize_tf is not None:
            video = self.resize_tf(video)
            video = video / 255.0
        # else: raw [0,255] float — pad/resize/normalize on GPU in LightningModule

        if self.use_um:
            video = apply_unsharp_masking(video, strength=self.um_strength)

        label = torch.tensor(label, dtype=torch.float32)

        return video, label
