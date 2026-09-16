"""
One-time preprocess: decode mp4 -> tensor cache on D: (keeps C: disk free).

  python scripts/build_video_cache.py --split train
  python scripts/build_video_cache.py --split all --workers 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from erdes.data.components.cached_video_dataset import cache_file_path
from erdes.data.components.erdes_dataset import VideoDataset

DEFAULT_CACHE_ROOT = Path("D:/ERDES/cache/non_rd_vs_rd")
DEFAULT_DATA_ROOT = Path("data/erdes")
SPLITS = {
    "train": root / "data/splits/non_rd_vs_rd/train.csv",
    "val": root / "data/splits/non_rd_vs_rd/val.csv",
    "test": root / "data/splits/non_rd_vs_rd/test.csv",
}


def _build_index(
    csv_path: str,
    cache_root: str,
    data_root: str,
    size: tuple[int, int, int],
    index: int,
) -> str:
    ds = VideoDataset(csv_path=csv_path, size=size, data_root=data_root)
    rel = ds.video_paths[index].replace("\\", "/")
    out = cache_file_path(Path(cache_root), rel)
    if out.is_file():
        return "skip"
    video, label = ds[index]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video.half(), "label": label}, out)
    return "ok"


def build_split(
    csv_path: Path,
    cache_root: Path,
    data_root: Path,
    size: tuple[int, int, int] = (96, 128, 128),
    workers: int = 1,
) -> tuple[int, int]:
    cache_root.mkdir(parents=True, exist_ok=True)
    ds = VideoDataset(csv_path=str(csv_path), size=size, data_root=str(data_root))
    n = len(ds)
    if workers <= 1:
        built = skipped = 0
        for i in tqdm(range(n), desc=csv_path.name):
            rel = ds.video_paths[i].replace("\\", "/")
            out = cache_file_path(cache_root, rel)
            if out.is_file():
                skipped += 1
                continue
            video, label = ds[i]
            torch.save({"video": video.half(), "label": label}, out)
            built += 1
        return built, skipped

    built = skipped = 0
    csv_s, cache_s, data_s = str(csv_path), str(cache_root), str(data_root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_build_index, csv_s, cache_s, data_s, size, i): i for i in range(n)
        }
        for fut in tqdm(as_completed(futures), total=n, desc=csv_path.name):
            if fut.result() == "skip":
                skipped += 1
            else:
                built += 1
    return built, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Parallel decode workers (Windows: 2-4).")
    args = parser.parse_args()

    splits = list(SPLITS.keys()) if args.split == "all" else [args.split]
    for name in splits:
        built, skipped = build_split(
            SPLITS[name], args.cache_root, args.data_root, workers=max(1, args.workers)
        )
        print(f"{name}: built={built}, skipped={skipped}")


if __name__ == "__main__":
    main()
