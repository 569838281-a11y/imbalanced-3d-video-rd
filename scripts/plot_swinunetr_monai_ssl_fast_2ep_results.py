"""Plot loss curves, PR curve, and confusion matrix for Swin-UNETR SSL 2-epoch run."""

from __future__ import annotations

import argparse
import json
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

from erdes.data.components.cached_video_dataset import CachedVideoDataset
from erdes.models.model_module import ModelModule
from erdes.training.threshold_search import (
    find_best_threshold_under_precision_constraint,
    plot_pr_curve,
)

DEFAULT_LOG = Path("D:/ERDES/logs/swinunetr_monai_ssl_fast_2ep_train.log")
DEFAULT_RUN = Path(
    "logs/train/runs/rd/swinunetr_monai_ssl_fast_2ep/2026-06-03_15-46-59"
)
DEFAULT_CKPT = (
    DEFAULT_RUN
    / "checkpoints/swinunetr_monai_ssl/swinunetr_monai_ssl_epoch_001.ckpt"
)
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv"
DEFAULT_CACHE = Path("D:/ERDES/cache/non_rd_vs_rd")


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_swin_log(log_path: Path) -> dict[str, list]:
    """Parse epoch-end train/val lines; keep last val block per epoch (post-train)."""
    text = re.sub(r"\s+", " ", read_log_text(log_path))
    val_pat = re.compile(
        r"Epoch (\d+): Val Loss: ([0-9.]+), Val Acc: ([0-9.]+), Val Precision: ([0-9.]+), "
        r"Val Sensitivity: ([0-9.]+), Val Specificity: ([0-9.]+), Val F1: ([0-9.]+)"
    )
    train_pat = re.compile(
        r"Epoch (\d+): Train Loss: ([0-9.]+), Train Acc: ([0-9.]+), Train Precision: ([0-9.]+), "
        r"Train Sensitivity: ([0-9.]+), Train Specificity: ([0-9.]+), Train F1: ([0-9.]+)"
    )
    val_rows: dict[int, dict[str, float]] = {}
    for m in val_pat.finditer(text):
        e = int(m.group(1))
        val_rows[e] = {
            "val_loss": float(m.group(2)),
            "val_accuracy": float(m.group(3)),
            "val_precision": float(m.group(4)),
            "val_sensitivity": float(m.group(5)),
            "val_specificity": float(m.group(6)),
            "val_f1": float(m.group(7)),
        }
    train_rows: dict[int, dict[str, float]] = {}
    for m in train_pat.finditer(text):
        e = int(m.group(1))
        train_rows[e] = {
            "train_loss": float(m.group(2)),
            "train_accuracy": float(m.group(3)),
            "train_precision": float(m.group(4)),
            "train_sensitivity": float(m.group(5)),
            "train_specificity": float(m.group(6)),
            "train_f1": float(m.group(7)),
        }
    epochs = sorted(set(val_rows) & set(train_rows))
    # Drop pre-train sanity val (epoch 0, acc=1, f1=0) if a post-train row exists
    if 0 in val_rows and val_rows[0].get("val_f1", 1) == 0 and 0 in train_rows:
        pass
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
        out[key] = []
    for e in epochs:
        row = {**val_rows.get(e, {}), **train_rows.get(e, {})}
        for key in keys:
            out[key].append(row.get(key, np.nan))
    return out


def plot_loss_curves(metrics: dict[str, list], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics["epochs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, metrics["train_loss"], "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs, metrics["val_loss"], "o-", label="Val Loss", linewidth=2)
    ax.set_title("MONAI SSL Swin-UNETR (2 epochs) — Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "loss_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


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
def eval_checkpoint(
    ckpt_path: Path,
    val_csv: Path,
    cache_dir: Path,
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
        cache_dir=str(cache_dir),
        strict_cache=True,
        use_um=False,
        cache_mmap=True,
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
    print(f"\n=== {name} (threshold={threshold:.4f}) ===")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.log.is_file():
        alt = PROJECT_ROOT / "logs" / "swinunetr_monai_ssl_fast_2ep_train.log"
        if alt.is_file():
            args.log = alt

    if not args.ckpt.exists():
        for candidate in [
            args.run_dir / "checkpoints/swinunetr_monai_ssl/last.ckpt",
            args.run_dir / "checkpoints/swinunetr_monai_ssl/swinunetr_monai_ssl_epoch_000.ckpt",
        ]:
            if candidate.exists():
                args.ckpt = candidate
                break
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing Swin 2-epoch training log ...")
    metrics = parse_swin_log(args.log)
    print(f"Epochs parsed: {metrics['epochs']}")

    print("\n--- Per-epoch metrics (from log, end of epoch) ---")
    print(f"{'Ep':>3} {'TLoss':>7} {'VLoss':>7} {'VAcc':>7} {'VF1':>7}")
    for i, e in enumerate(metrics["epochs"]):
        print(
            f"{e:3d} {metrics['train_loss'][i]:7.3f} {metrics['val_loss'][i]:7.3f} "
            f"{metrics['val_accuracy'][i]:7.3f} {metrics['val_f1'][i]:7.3f}"
        )

    loss_path = plot_loss_curves(metrics, plots_dir)
    print(f"Saved: {loss_path}")

    print(f"\nEvaluating checkpoint on val set: {args.ckpt}")
    y_true, y_prob, y_pred = eval_checkpoint(
        args.ckpt,
        args.val_csv,
        args.cache_dir,
        args.batch_size,
        args.device,
    )
    m05 = binary_metrics(y_true, y_pred, y_prob)
    print_metrics_table("Swin SSL 2ep (epoch_001 ckpt, thr=0.5)", m05, 0.5)

    thr_res = find_best_threshold_under_precision_constraint(y_true, y_prob, min_precision=0.5)
    y_pred_opt = (y_prob >= thr_res["threshold"]).astype(np.int64)
    m_opt = binary_metrics(y_true, y_pred_opt, y_prob)
    print_metrics_table("Swin SSL 2ep (PR-optimized thr)", m_opt, thr_res["threshold"])

    plot_confusion(
        np.array(m05["confusion_matrix"]),
        plots_dir / "confusion_matrix_val_thr0.5.png",
        "Swin-UNETR SSL (2ep) — Confusion Matrix (thr=0.5)",
    )
    plot_confusion(
        np.array(m_opt["confusion_matrix"]),
        plots_dir / "confusion_matrix_val_pr_opt.png",
        f"Swin-UNETR SSL (2ep) — Confusion Matrix (thr={thr_res['threshold']:.3f})",
    )
    plot_pr_curve(y_true, y_prob, plots_dir / "pr_curve_val.png", thr_res["threshold"])
    print(f"Saved: {plots_dir / 'pr_curve_val.png'}")
    print(f"Saved: {plots_dir / 'confusion_matrix_val_thr0.5.png'}")
    print(f"Saved: {plots_dir / 'confusion_matrix_val_pr_opt.png'}")

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
