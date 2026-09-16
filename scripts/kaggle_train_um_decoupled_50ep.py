"""
Kaggle entry: UM + Decoupled ResNet3D, 50 epochs, detailed recall metrics.

Usage on Kaggle (after uploading Datasets + this repo to /kaggle/working/test_5):

  %cd /kaggle/working/test_5
  !python scripts/kaggle_train_um_decoupled_50ep.py

Or with path overrides:

  !python scripts/kaggle_train_um_decoupled_50ep.py \\
      --um-cache /kaggle/input/YOUR_DS/non_rd_vs_rd_um \\
      --base-cache /kaggle/input/YOUR_DS/non_rd_vs_rd \\
      --splits /kaggle/input/YOUR_SPLITS/non_rd_vs_rd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UM+Decoupled 50ep on Kaggle")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing erdes/ and configs/",
    )
    p.add_argument(
        "--um-cache",
        type=Path,
        default=Path("/kaggle/input/erdes-um-cache/non_rd_vs_rd_um"),
    )
    p.add_argument(
        "--base-cache",
        type=Path,
        default=Path("/kaggle/input/erdes-um-cache/non_rd_vs_rd"),
    )
    p.add_argument(
        "--splits",
        type=Path,
        default=Path("/kaggle/input/erdes-splits/non_rd_vs_rd"),
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument(
        "--experiment",
        default="non_rd_vs_rd/resnet3d_um_decoupled_50ep",
        help="Hydra experiment config name",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    os.chdir(root)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_csv = args.splits / "train.csv"
    val_csv = args.splits / "val.csv"
    test_csv = args.splits / "test.csv"

    for path, label in [
        (args.um_cache, "UM cache"),
        (train_csv, "train.csv"),
        (val_csv, "val.csv"),
    ]:
        if not path.exists():
            print(f"[ERROR] Missing {label}: {path}")
            print("Upload Datasets and pass --um-cache / --splits as needed.")
            return 1

    um_count = len(list(args.um_cache.glob("*.pt")))
    print(f"[INFO] Repo: {root}")
    print(f"[INFO] UM cache: {args.um_cache} ({um_count} .pt files)")
    print(f"[INFO] Base cache: {args.base_cache} (exists={args.base_cache.is_dir()})")
    print(f"[INFO] Splits: {args.splits}")
    print(f"[INFO] Experiment: {args.experiment}, max_epochs={args.max_epochs}")
    print("[INFO] Metrics: val/recall, multi-thr recall, TP/FP/FN/TN → epoch_metrics.csv")

    cmd = [
        sys.executable,
        "-u",
        str(root / "erdes" / "train.py"),
        f"experiment={args.experiment}",
        f"trainer.max_epochs={args.max_epochs}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        f"data.um_cache_dir={args.um_cache.as_posix()}",
        f"data.cache_dir={args.base_cache.as_posix()}",
        f"data.train_csv={train_csv.as_posix()}",
        f"data.val_csv={val_csv.as_posix()}",
        f"data.test_csv={test_csv.as_posix()}",
        "model.compute_train_metrics=true",
        "model.gpu_preprocess=false",
        "tags=[erdes,um,decoupled,50ep,kaggle]",
    ]
    print("[CMD]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
