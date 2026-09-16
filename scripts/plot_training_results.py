"""Plot loss curves and confusion matrix from a resnet3d training run."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from erdes.data.components.erdes_dataset import VideoDataset
from erdes.models.model_module import ModelModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "logs/train/runs/rd/resnet3d/2026-05-29_22-02-10"
DEFAULT_LOG = PROJECT_ROOT / "logs/latest_train_output.log"
DEFAULT_CKPT = (
    DEFAULT_RUN_DIR / "checkpoints/resnet3d/resnet3d_best_epoch_015.ckpt"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/erdes"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv"


def read_log_text(log_path: Path) -> str:
    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="ignore")


def parse_log_metrics(log_path: Path) -> dict[str, list[float]]:
    text = read_log_text(log_path)
    pattern = re.compile(
        r"Epoch (\d+): (Train|Val) Loss: ([0-9.]+), (?:Train|Val) Acc: ([0-9.]+)"
    )
    records: dict[int, dict[str, float]] = {}
    for epoch_str, split, loss_str, acc_str in pattern.findall(text):
        epoch = int(epoch_str)
        records.setdefault(epoch, {})
        records[epoch][f"{split.lower()}_loss"] = float(loss_str)
        records[epoch][f"{split.lower()}_acc"] = float(acc_str)

    # Drop the pre-training sanity-check validation block at epoch 0.
    if 0 in records and records[0].get("val_acc", 1.0) == 0.0:
        del records[0]

    epochs = sorted(records)
    return {
        "epochs": epochs,
        "train_loss": [records[e]["train_loss"] for e in epochs],
        "val_loss": [records[e]["val_loss"] for e in epochs],
        "train_acc": [records[e]["train_acc"] for e in epochs],
        "val_acc": [records[e]["val_acc"] for e in epochs],
    }


def plot_loss_curves(metrics: dict[str, list[float]], output_dir: Path) -> Path:
    epochs = metrics["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, metrics["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, metrics["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss Curves (Non-RD vs RD, ResNet3D)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, metrics["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs, metrics["val_acc"], marker="o", label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = output_dir / "loss_and_acc_curves.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.inference_mode()
def collect_predictions(
    ckpt_path: Path,
    val_csv: Path,
    data_root: Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = ModelModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)

    dataset = VideoDataset(
        csv_path=str(val_csv),
        size=(96, 128, 128),
        data_root=str(data_root),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        logits, batch_y = model.prepare_for_bce_loss(logits, batch_y.to(device))
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int().view(-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(batch_y.int().view(-1).cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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
    ax.set_title("Confusion Matrix (Validation Set, Best Checkpoint)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    out_path = output_dir / "confusion_matrix_val.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-confusion", action="store_true")
    args = parser.parse_args()

    output_dir = args.run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_log_metrics(args.log_path)
    curve_path = plot_loss_curves(metrics, output_dir)
    print(f"Saved: {curve_path}")

    if not args.skip_confusion:
        if not args.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        print(f"Running validation inference on {args.device} ...")
        y_true, y_pred = collect_predictions(
            ckpt_path=args.ckpt_path,
            val_csv=args.val_csv,
            data_root=args.data_root,
            batch_size=args.batch_size,
            device=args.device,
        )
        cm_path = plot_confusion_matrix(y_true, y_pred, output_dir)
        print(f"Saved: {cm_path}")
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
