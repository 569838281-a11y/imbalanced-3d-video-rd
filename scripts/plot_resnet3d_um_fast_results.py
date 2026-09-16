"""Plot loss/metrics/CM/PR for resnet3d_um_fast and UM before-after samples."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from erdes.data.components.cached_video_dataset import CachedVideoDataset, cache_file_path
from erdes.models.model_module import ModelModule
from erdes.training.threshold_search import (
    find_best_threshold_under_precision_constraint,
    plot_pr_curve,
)

DEFAULT_LOG = Path("D:/ERDES/logs/resnet3d_um_fast_train.log")
DEFAULT_RUN = Path("D:/ERDES/logs/train/runs/rd/resnet3d_um_fast/2026-06-02_20-01-30")
DEFAULT_CKPT = DEFAULT_RUN / "checkpoints/resnet3d_um/resnet3d_um_fast_epoch_012.ckpt"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv"
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data/splits/non_rd_vs_rd/train.csv"
DEFAULT_BASE_CACHE = Path("D:/ERDES/cache/non_rd_vs_rd")
DEFAULT_UM_CACHE = Path("D:/ERDES/cache/non_rd_vs_rd_um")


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_um_fast_log(log_path: Path) -> dict[str, list]:
    text = re.sub(r"\s+", " ", read_log_text(log_path))
    val_pat = re.compile(
        r"Epoch (\d+): Val Loss: ([0-9.]+), Val Acc: ([0-9.]+), Val Precision: ([0-9.]+), "
        r"Val Sensitivity: ([0-9.]+), Val Specificity: ([0-9.]+), Val F1: ([0-9.]+)"
    )
    train_pat = re.compile(
        r"Epoch (\d+): Train Loss: ([0-9.]+), Train Acc: ([0-9.]+), Train Precision: ([0-9.]+), "
        r"Train Sensitivity: ([0-9.]+), Train Specificity: ([0-9.]+), Train F1: ([0-9.]+)"
    )
    rows: dict[int, dict[str, float]] = {}
    for m in val_pat.finditer(text):
        e = int(m.group(1))
        acc = float(m.group(3))
        if e == 0 and acc <= 0.2:
            continue
        rows[e] = {
            "val_loss": float(m.group(2)),
            "val_accuracy": acc,
            "val_precision": float(m.group(4)),
            "val_sensitivity": float(m.group(5)),
            "val_specificity": float(m.group(6)),
            "val_f1": float(m.group(7)),
        }
    for m in train_pat.finditer(text):
        e = int(m.group(1))
        if e not in rows:
            rows[e] = {}
        rows[e].update(
            {
                "train_loss": float(m.group(2)),
                "train_accuracy": float(m.group(3)),
                "train_precision": float(m.group(4)),
                "train_sensitivity": float(m.group(5)),
                "train_specificity": float(m.group(6)),
                "train_f1": float(m.group(7)),
            }
        )
    epochs = sorted(rows)
    out: dict[str, list] = {"epochs": epochs}
    keys = (
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_precision",
        "val_precision",
        "train_sensitivity",
        "val_sensitivity",
        "train_specificity",
        "val_specificity",
        "train_f1",
        "val_f1",
    )
    for key in keys:
        out[key] = [rows[e].get(key, np.nan) for e in epochs]
    return out


def plot_loss_and_metrics(metrics: dict[str, list], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics["epochs"]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, metrics["train_loss"], "o-", label="Train Loss")
    ax.plot(epochs, metrics["val_loss"], "o-", label="Val Loss")
    ax.set_title("ResNet3D + UM (fast) — Loss Curves (15 epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = output_dir / "loss_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("val_accuracy", "train_accuracy", "Accuracy"),
        ("val_precision", "train_precision", "Precision"),
        ("val_sensitivity", "train_sensitivity", "Sensitivity"),
        ("val_specificity", "train_specificity", "Specificity"),
        ("val_f1", "train_f1", "F1"),
    ]
    for ax, (vk, tk, title) in zip(axes.flat[:5], panels):
        ax.plot(epochs, metrics[vk], "o-", label=f"Val {title}")
        ax.plot(epochs, metrics[tk], "o-", label=f"Train {title}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes.flat[5].axis("off")
    fig.suptitle("ResNet3D + UM (fast) — Metrics per Epoch", y=1.02)
    fig.tight_layout()
    p = output_dir / "metric_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


@torch.inference_mode()
def eval_um_cached_checkpoint(
    ckpt_path: Path,
    val_csv: Path,
    um_cache_dir: Path,
    batch_size: int,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = ModelModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.float()
    model.to(device)

    dataset = CachedVideoDataset(
        csv_path=str(val_csv),
        size=(96, 128, 128),
        cache_dir=str(um_cache_dir),
        strict_cache=True,
        use_um=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.float().to(device)
        logits = model(batch_x)
        logits, batch_y = model.prepare_for_bce_loss(logits, batch_y.to(device).float())
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs_list.append(probs)
        labels_list.append(batch_y.int().cpu().numpy().reshape(-1))

    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return y_true, y_prob, y_pred


def plot_confusion(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Non-RD", "Pred RD"],
        yticklabels=["True Non-RD", "True RD"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_metrics_table(name: str, metrics: dict, threshold: float) -> None:
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


def load_cached_tensor(cache_root: Path, rel_path: str) -> torch.Tensor:
    path = cache_file_path(cache_root, rel_path.replace("\\", "/"))
    obj = torch.load(path, map_location="cpu", weights_only=True)
    video = obj["video"] if isinstance(obj, dict) else obj
    return video.float()


def frame_for_display(video: torch.Tensor, frame_idx: int | None = None) -> np.ndarray:
    """Return HxW grayscale from [C,D,H,W]."""
    v = video.numpy()
    if frame_idx is None:
        frame_idx = v.shape[1] // 2
    if v.shape[0] >= 3:
        frame = v[:3, frame_idx].transpose(1, 2, 0)
        frame = frame.mean(axis=2)
    else:
        frame = v[0, frame_idx]
    hi = 255.0 if frame.max() > 1.5 else 1.0
    frame = np.clip(frame, 0, hi) / hi
    return frame


def plot_um_comparisons(
    train_csv: Path,
    base_cache: Path,
    um_cache: Path,
    output_dir: Path,
    n_samples: int = 10,
    seed: int = 42,
) -> Path:
    import pandas as pd

    df = pd.read_csv(train_csv)
    paths = df["path"].tolist()
    labels = df["label"].tolist()
    rng = random.Random(seed)
    indices = rng.sample(range(len(paths)), min(n_samples, len(paths)))

    n_cols = 2
    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        rel = paths[idx].replace("\\", "/")
        label = "RD" if int(labels[idx]) == 1 else "Non-RD"
        orig = load_cached_tensor(base_cache, rel)
        sharp = load_cached_tensor(um_cache, rel)
        mid = orig.shape[1] // 2
        axes[row, 0].imshow(frame_for_display(orig, mid), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Original (t={mid}) — {label}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(frame_for_display(sharp, mid), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"UM sharpened (t={mid})")
        axes[row, 1].axis("off")
        short = rel if len(rel) <= 48 else "..." + rel[-45:]
        fig.text(0.02, 1.0 - (row + 0.5) / n_rows, short, fontsize=7, va="center")

    fig.suptitle("Original vs Unsharp Masking (10 random train samples)", y=1.002, fontsize=12)
    fig.tight_layout()
    out = output_dir / "um_comparison_10_samples.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--um-cache", type=Path, default=DEFAULT_UM_CACHE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-um-samples", type=int, default=10)
    args = parser.parse_args()

    if not args.ckpt.exists():
        alt = args.run_dir / "checkpoints/resnet3d_um/last.ckpt"
        if alt.exists():
            args.ckpt = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing UM fast training log ...")
    metrics = parse_um_fast_log(args.log)
    print(f"Epochs parsed: {metrics['epochs']}")

    print("\n--- Per-epoch validation metrics (from log) ---")
    print(
        f"{'Ep':>3} {'Loss':>7} {'Acc':>7} {'Prec':>7} {'Sens':>7} {'Spec':>7} {'F1':>7}"
    )
    for i, e in enumerate(metrics["epochs"]):
        print(
            f"{e:3d} {metrics['val_loss'][i]:7.3f} {metrics['val_accuracy'][i]:7.3f} "
            f"{metrics['val_precision'][i]:7.3f} {metrics['val_sensitivity'][i]:7.3f} "
            f"{metrics['val_specificity'][i]:7.3f} {metrics['val_f1'][i]:7.3f}"
        )

    for p in plot_loss_and_metrics(metrics, plots_dir):
        print(f"Saved: {p}")

    print(f"\nEvaluating checkpoint: {args.ckpt}")
    y_true, y_prob, y_pred = eval_um_cached_checkpoint(
        args.ckpt,
        args.val_csv,
        args.um_cache,
        args.batch_size,
        args.device,
    )
    m05 = binary_metrics(y_true, y_pred, y_prob)
    print_metrics_table("ResNet3D+UM fast (best ckpt, thr=0.5)", m05, 0.5)

    thr_res = find_best_threshold_under_precision_constraint(y_true, y_prob, min_precision=0.5)
    y_pred_opt = (y_prob >= thr_res["threshold"]).astype(np.int64)
    m_opt = binary_metrics(y_true, y_pred_opt, y_prob)
    print_metrics_table("ResNet3D+UM fast (PR-optimized thr)", m_opt, thr_res["threshold"])

    plot_confusion(
        np.array(m05["confusion_matrix"]),
        plots_dir / "confusion_matrix_val_thr0.5.png",
        "ResNet3D+UM (fast) — Confusion Matrix (thr=0.5)",
    )
    plot_pr_curve(y_true, y_prob, plots_dir / "pr_curve_val.png", thr_res["threshold"])
    print(f"Saved: {plots_dir / 'pr_curve_val.png'}")
    print(f"Saved: {plots_dir / 'confusion_matrix_val_thr0.5.png'}")

    cmp_path = plot_um_comparisons(
        args.train_csv,
        args.base_cache,
        args.um_cache,
        plots_dir,
        n_samples=args.n_um_samples,
    )
    print(f"Saved: {cmp_path}")

    summary = {
        "per_epoch_log": metrics,
        "checkpoint_eval_thr_0.5": m05,
        "checkpoint_eval_pr_optimized": {**m_opt, "threshold": thr_res["threshold"]},
        "checkpoint": str(args.ckpt),
    }
    summary_path = plots_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
