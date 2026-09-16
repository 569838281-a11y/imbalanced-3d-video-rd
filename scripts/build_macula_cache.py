"""Build .pt cache for macula_detached_vs_intact splits."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from erdes.data.components.cached_video_dataset import cache_file_path
from erdes.data.components.erdes_dataset import VideoDataset

CACHE_ROOT = Path("D:/ERDES/cache/macula_detached_vs_intact")
DATA_ROOT = root / "data/erdes"
SIZE = (96, 128, 128)
SPLITS = {
    "train": root / "data/splits/macula_detached_vs_intact/train.csv",
    "val": root / "data/splits/macula_detached_vs_intact/val.csv",
    "test": root / "data/splits/macula_detached_vs_intact/test.csv",
}


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    built = skipped = failed = 0
    for name, csv_path in SPLITS.items():
        ds = VideoDataset(csv_path=str(csv_path), size=SIZE, data_root=str(DATA_ROOT))
        for i in tqdm(range(len(ds)), desc=name):
            rel = ds.video_paths[i].replace("\\", "/")
            out = cache_file_path(CACHE_ROOT, rel)
            if out.is_file():
                skipped += 1
                continue
            try:
                video, label = ds[i]
                torch.save({"video": video.half(), "label": label}, out)
                built += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {rel}: {exc}")
    n = len(list(CACHE_ROOT.glob("*.pt")))
    print(f"done built={built} skipped={skipped} failed={failed} total_pt={n}")


if __name__ == "__main__":
    main()
