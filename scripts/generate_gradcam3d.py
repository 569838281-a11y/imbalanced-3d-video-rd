"""
Generate stratified Grad-CAM heatmaps for ResNet3D (15-epoch base checkpoint).

Example:
  python scripts/generate_gradcam3d.py
  python scripts/generate_gradcam3d.py --num-samples 500 --output-dir GradCAM
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from erdes.data.components import erdes_dataset as eds
from erdes.models.model_module import ModelModule
from erdes.visualization.gradcam3d import GradCAM3D, save_heatmap_png

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (
    PROJECT_ROOT
    / "logs/train/runs/rd/resnet3d/2026-05-29_22-02-10/checkpoints/resnet3d/resnet3d_best_epoch_015.ckpt"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/erdes"
DEFAULT_SPLITS = PROJECT_ROOT / "data/splits/non_rd_vs_rd"
DEFAULT_OUTPUT = PROJECT_ROOT / "GradCAM"


def load_all_videos(splits_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = splits_dir / name
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No CSV splits found under {splits_dir}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["path"])
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample n_total rows preserving dataset label ratio (with replacement if needed)."""
    ratio = float(df["label"].mean())
    n_pos = int(round(n_total * ratio))
    n_neg = n_total - n_pos
    if n_total >= 2:
        if n_pos == 0:
            n_pos = 1
            n_neg = n_total - 1
        elif n_neg == 0:
            n_neg = 1
            n_pos = n_total - 1

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError("Both classes must be present in the dataset pool.")

    replace_pos = len(pos_df) < n_pos
    replace_neg = len(neg_df) < n_neg
    if replace_pos or replace_neg:
        print(
            f"Warning: class count smaller than quota; sampling with replacement "
            f"(pos={n_pos}, neg={n_neg})."
        )

    pos_sample = pos_df.sample(n=n_pos, replace=replace_pos, random_state=seed)
    neg_sample = neg_df.sample(n=n_neg, replace=replace_neg, random_state=seed + 1)
    out = pd.concat([neg_sample, pos_sample], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def build_filename_counters() -> dict[int, int]:
    return {0: 0, 1: 0}


def next_filename(label: int, counters: dict[int, int]) -> str:
    counters[label] += 1
    prefix = "rd" if int(label) == 1 else "non_rd"
    return f"{prefix}_{counters[label]:03d}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D Grad-CAM heatmaps for ResNet3D.")
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-index", type=int, default=-1, help="Temporal index; -1 = middle frame.")
    args = parser.parse_args()

    if not args.ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt_path}\n"
            "Train base resnet3d for 15 epochs or pass --ckpt-path."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_all_videos(args.splits_dir)
    sampled = stratified_sample(pool, args.num_samples, args.seed)
    dataset_ratio = float(pool["label"].mean())
    sample_ratio = float(sampled["label"].mean())

    print(
        f"Pool size: {len(pool)} | sample RD ratio: {sample_ratio:.4f} "
        f"(dataset ~{dataset_ratio:.4f})"
    )
    print(f"Loading checkpoint: {args.ckpt_path}")

    model = ModelModule.load_from_checkpoint(str(args.ckpt_path), map_location=args.device)
    model.eval()
    model.to(args.device)
    gradcam = GradCAM3D(model.net)

    from erdes.data.components.utils import resize

    resize_tf = resize((96, 128, 128))

    counters = build_filename_counters()
    manifest: list[dict] = []
    frame_index = None if args.frame_index < 0 else args.frame_index

    try:
        for row in tqdm(sampled.itertuples(index=False), total=len(sampled), desc="GradCAM"):
            rel_path = row.path
            label = int(row.label)
            full_path = os.path.join(str(args.data_root), rel_path.replace("\\", "/"))
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Video not found: {full_path}")

            depth = 96
            raw = eds._read_video_frames(str(full_path), num_frames=depth, max_decode=max(depth * 2, 160))
            video = raw.float().permute(3, 0, 1, 2)
            if video.shape[0] == 3:
                video = video.mean(dim=0, keepdim=True)
            video = resize_tf(video)
            video = video / 255.0

            batch = video.unsqueeze(0).to(args.device)
            cam = gradcam(batch, label=label)

            fname = next_filename(label, counters)
            out_path = args.output_dir / fname
            save_heatmap_png(video, cam, out_path, frame_index=frame_index)

            manifest.append(
                {
                    "filename": fname,
                    "path": rel_path,
                    "label": label,
                    "class_name": "rd" if label == 1 else "non_rd",
                }
            )
    finally:
        gradcam.close()

    meta_path = args.output_dir / "gradcam_manifest.json"
    summary = {
        "checkpoint": str(args.ckpt_path),
        "num_samples": len(manifest),
        "dataset_rd_ratio": dataset_ratio,
        "sample_rd_ratio": sample_ratio,
        "output_dir": str(args.output_dir),
        "entries": manifest,
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} heatmaps to {args.output_dir}")
    print(f"Manifest: {meta_path}")


if __name__ == "__main__":
    main()
