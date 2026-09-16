"""
Self-supervised VideoMAE pretraining for Swin-UNETR backbone (15 epochs).

  python run_pretrain_swinunetr.py
  python run_pretrain_swinunetr.py --epochs 15 --batch-size 2 --device cuda

Requires D: video cache (see scripts/build_video_cache.py).
Saves encoder weights to weights/swinunetr_videomae_backbone.pth
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from erdes.data.components.cached_video_dataset import CachedVideoDataset
from erdes.models.components.swinunetr_videomae_pretrain import SwinUNETRVideoMAEPretrain

DEFAULT_CACHE = Path("D:/ERDES/cache/non_rd_vs_rd")
TRAIN_ONLY_SPLITS = [PROJECT_ROOT / "data/splits/non_rd_vs_rd/train.csv"]
DEFAULT_SPLITS = [
    PROJECT_ROOT / "data/splits/non_rd_vs_rd/train.csv",
    PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv",
]
DEFAULT_SIZE = (96, 128, 128)
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights/swinunetr_videomae_backbone.pth"
DEFAULT_PROGRESS_LOG = Path("D:/ERDES/logs/pretrain_swinunetr_progress.log")


class UnlabeledVideoDataset(Dataset):
    """Unlabeled ultrasound videos from preprocessed D: cache (no labels)."""

    def __init__(
        self,
        csv_paths: list[Path],
        cache_dir: Path,
        size: tuple[int, int, int] = DEFAULT_SIZE,
    ) -> None:
        paths: list[str] = []
        for csv in csv_paths:
            df = pd.read_csv(csv)
            paths.extend(df["path"].astype(str).tolist())
        self.paths = paths
        self.size = size
        self.cache_dir = str(cache_dir)
        self._ds = CachedVideoDataset(
            csv_path=str(csv_paths[0]),
            size=size,
            cache_dir=self.cache_dir,
            strict_cache=True,
            use_um=False,
        )
        self._ds.video_paths = self.paths
        self._ds.labels = [0] * len(self.paths)
        self._ds._fallback.video_paths = self.paths
        self._ds._fallback.labels = self._ds.labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        video, _ = self._ds[idx]
        return video


def _append_progress(log_path: Path | None, msg: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train_one_epoch(
    model: SwinUNETRVideoMAEPretrain,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    log_every: int,
    global_step: int,
    progress_log: Path | None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    n_steps = len(loader)

    print(
        f"\n[Train] Epoch {epoch - 1} started — {n_steps} steps "
        f"(log every {log_every} steps)\n",
        flush=True,
    )

    for batch_idx, video in enumerate(loader):
        video = video.to(device, non_blocking=True)
        if video.ndim == 4:
            video = video.unsqueeze(1)
        video = video.float()

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, _, _ = model(video)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, _ = model(video)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        global_step += 1

        if log_every > 0 and global_step % log_every == 0:
            avg = total_loss / n_batches
            step_loss = float(loss.item())
            msg = (
                f"[step {global_step}] epoch {epoch - 1} "
                f"batch {batch_idx + 1}/{n_steps} "
                f"loss(step)={step_loss:.6f} loss(avg)={avg:.6f} "
                f"elapsed={time.time() - t0:.1f}s"
            )
            print(msg, flush=True)
            _append_progress(progress_log, msg)

    return total_loss / max(n_batches, 1), global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Swin-UNETR VideoMAE pretraining")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--splits", type=str, nargs="+", default=[str(p) for p in DEFAULT_SPLITS])
    parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N steps.")
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=DEFAULT_PROGRESS_LOG,
        help="Append step progress lines to this file.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all cached videos")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Pretrain on train split only (~10%% fewer steps).",
    )
    parser.add_argument("--decoder-layers", type=int, default=2, help="MAE decoder Transformer layers (default 2).")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Gradient checkpointing in Swin encoder (saves VRAM, may allow larger batch).",
    )
    parser.add_argument(
        "--no-fast-loss",
        action="store_true",
        help="Disable patch-level MSE (slower, assembles full volume).",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (PyTorch 2+).")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train_only:
        args.splits = [str(p) for p in TRAIN_ONLY_SPLITS]

    cache_dir = args.cache_dir
    if not cache_dir.is_dir() or len(list(cache_dir.glob("*.pt"))) < 100:
        print(f"Cache missing or too small: {cache_dir}")
        print("Run: python scripts/build_video_cache.py --split all --workers 4")
        sys.exit(1)

    csv_paths = [Path(p) for p in args.splits]
    dataset = UnlabeledVideoDataset(csv_paths, cache_dir, DEFAULT_SIZE)
    if args.max_samples > 0:
        dataset.paths = dataset.paths[: args.max_samples]
        dataset._ds.video_paths = dataset.paths
        dataset._ds.labels = [0] * len(dataset.paths)
        dataset._ds._fallback.video_paths = dataset.paths
        dataset._ds._fallback.labels = dataset._ds.labels

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = SwinUNETRVideoMAEPretrain(
        img_size=DEFAULT_SIZE,
        in_channels=1,
        mask_ratio=args.mask_ratio,
        decoder_layers=args.decoder_layers,
        use_checkpoint=args.use_checkpoint,
        patch_level_loss=not args.no_fast_loss,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    print(f"Token shape (bottleneck): {model.token_shape}")
    print(
        f"fast_loss={not args.no_fast_loss} | decoder_layers={args.decoder_layers} | "
        f"use_checkpoint={args.use_checkpoint} | compile={args.compile}",
        flush=True,
    )
    print(f"Tube mask generator: {model.mask_generator}")
    steps_per_epoch = len(loader)
    print(
        f"Dataset size: {len(dataset)} videos | batch_size={args.batch_size} | "
        f"steps/epoch={steps_per_epoch} | log_every={args.log_every}",
        flush=True,
    )

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====", flush=True)
        avg_loss, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            scaler,
            epoch + 1,
            args.log_every,
            global_step,
            args.progress_log,
        )
        print(f"Epoch {epoch + 1} mean reconstruction loss: {avg_loss:.6f}", flush=True)

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), args.weights_out)
    print(f"\nSaved Swin-UNETR encoder weights -> {args.weights_out.resolve()}")


if __name__ == "__main__":
    main()
