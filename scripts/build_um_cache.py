"""
Offline Unsharp Masking: read decoded tensors from D: cache, write UM cache.

Prerequisite: python scripts/build_video_cache.py --split all

  python scripts/build_um_cache.py --split train --workers 4
  python scripts/build_um_cache.py --split all --workers 4 --strength 1.5
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
from erdes.data.components.unsharp_masking import apply_unsharp_masking

DEFAULT_SRC = Path("D:/ERDES/cache/non_rd_vs_rd")
DEFAULT_DST = Path("D:/ERDES/cache/non_rd_vs_rd_um")
DEFAULT_DATA_ROOT = Path("data/erdes")
SPLITS = {
    "train": root / "data/splits/non_rd_vs_rd/train.csv",
    "val": root / "data/splits/non_rd_vs_rd/val.csv",
    "test": root / "data/splits/non_rd_vs_rd/test.csv",
}


def _process_one(
    rel: str,
    src_root: str,
    dst_root: str,
    strength: float,
) -> str:
    src = cache_file_path(Path(src_root), rel)
    dst = cache_file_path(Path(dst_root), rel)
    if dst.is_file():
        return "skip"
    if not src.is_file():
        return "miss"
    obj = torch.load(src, map_location="cpu", weights_only=True)
    video = obj["video"] if isinstance(obj, dict) else obj
    label = obj.get("label") if isinstance(obj, dict) else None
    video_um = apply_unsharp_masking(video.float(), strength=strength)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video": video_um.half(), "label": label}
    torch.save(payload, dst)
    return "ok"


def build_split(
    csv_path: Path,
    src_root: Path,
    dst_root: Path,
    data_root: Path,
    strength: float = 1.5,
    workers: int = 1,
) -> tuple[int, int, int]:
    dst_root.mkdir(parents=True, exist_ok=True)
    ds = VideoDataset(csv_path=str(csv_path), size=(96, 128, 128), data_root=str(data_root))
    rels = [p.replace("\\", "/") for p in ds.video_paths]
    n = len(rels)

    if workers <= 1:
        built = skipped = missed = 0
        for rel in tqdm(rels, desc=csv_path.name):
            status = _process_one(rel, str(src_root), str(dst_root), strength)
            if status == "skip":
                skipped += 1
            elif status == "miss":
                missed += 1
            else:
                built += 1
        return built, skipped, missed

    built = skipped = missed = 0
    src_s, dst_s = str(src_root), str(dst_root)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, rel, src_s, dst_s, strength): rel for rel in rels
        }
        for fut in tqdm(as_completed(futures), total=n, desc=csv_path.name):
            status = fut.result()
            if status == "skip":
                skipped += 1
            elif status == "miss":
                missed += 1
            else:
                built += 1
    return built, skipped, missed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "train_val", "test", "all"],
        default="train_val",
        help="train_val = train+val only (matches UM training); test usually skipped.",
    )
    parser.add_argument("--src-cache", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst-cache", type=Path, default=DEFAULT_DST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.split == "train_val":
        splits = ["train", "val"]
    elif args.split == "all":
        splits = list(SPLITS.keys())
    else:
        splits = [args.split]

    for name in splits:
        built, skipped, missed = build_split(
            SPLITS[name],
            args.src_cache,
            args.dst_cache,
            args.data_root,
            strength=args.strength,
            workers=max(1, args.workers),
        )
        print(f"{name}: built={built}, skipped={skipped}, missed={missed}")


if __name__ == "__main__":
    main()
