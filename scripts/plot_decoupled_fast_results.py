"""Plot metrics / CM / PR for decoupled_fast run and baseline resnet3d PR curve."""

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
from erdes.models.components.factory import build_3d_architecture
from erdes.models.decoupled_model_module import DecoupledModelModule
from erdes.models.model_module import ModelModule
from erdes.training.threshold_search import (
    find_best_threshold_under_precision_constraint,
    plot_pr_curve,
)

DEFAULT_FAST_LOG = Path("D:/ERDES/logs/decoupled_fast_train.log")
DEFAULT_FAST_RUN = (
    PROJECT_ROOT
    / "logs/train/runs/rd/resnet3d_decoupled_fast/2026-06-02_11-02-26"
)
DEFAULT_FAST_CKPT = (
    DEFAULT_FAST_RUN / "checkpoints/resnet3d_decoupled/decoupled_fast_epoch_010.ckpt"
)
DEFAULT_BASE_RUN = PROJECT_ROOT / "logs/train/runs/rd/resnet3d/2026-05-29_22-02-10"
DEFAULT_BASE_CKPT = (
    DEFAULT_BASE_RUN / "checkpoints/resnet3d/resnet3d_best_epoch_015.ckpt"
)
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv"
DEFAULT_CACHE_DIR = Path("D:/ERDES/cache/non_rd_vs_rd")


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_decoupled_log(log_path: Path) -> dict[str, list]:
    text = re.sub(r"\s+", " ", read_log_text(log_path))
    val_pat = re.compile(
        r"Epoch (\d+) \[Stage \d+\]: Val Loss: ([0-9.]+), Val F1: ([0-9.]+), "
        r"Val PR-AUC: ([0-9.]+), Val Sensitivity: ([0-9.]+), Val Precision: ([0-9.]+)"
    )
    train_pat = re.compile(
        r"Epoch (\d+) \[Stage \d+\]: Train Loss: ([0-9.]+), Train F1: ([0-9.]+), "
        r"Train Sensitivity: ([0-9.]+), Train Precision: ([0-9.]+)"
    )
    val_rows = {}
    for m in val_pat.finditer(text):
        e = int(m.group(1))
        val_rows[e] = {
            "val_loss": float(m.group(2)),
            "val_f1": float(m.group(3)),
            "val_pr_auc": float(m.group(4)),
            "val_sensitivity": float(m.group(5)),
            "val_precision": float(m.group(6)),
        }
    for m in train_pat.finditer(text):
        e = int(m.group(1))
        if e not in val_rows:
            val_rows[e] = {}
        val_rows[e].update(
            {
                "train_loss": float(m.group(2)),
                "train_f1": float(m.group(3)),
                "train_sensitivity": float(m.group(4)),
                "train_precision": float(m.group(5)),
            }
        )
    epochs = sorted(val_rows)
    out: dict[str, list] = {"epochs": epochs}
    for key in (
        "train_loss",
        "val_loss",
        "train_f1",
        "val_f1",
        "train_precision",
        "val_precision",
        "train_sensitivity",
        "val_sensitivity",
        "val_pr_auc",
    ):
        out[key] = [val_rows[e].get(key, np.nan) for e in epochs]
    return out


def plot_metric_curves(metrics: dict[str, list], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = metrics["epochs"]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, metrics["train_loss"], "o-", label="Train Loss")
    ax.plot(epochs, metrics["val_loss"], "o-", label="Val Loss")
    ax.set_title("Decoupled Fast — Loss Curves (15 epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = output_dir / "loss_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("val_f1", "train_f1", "F1"),
        ("val_precision", "train_precision", "Precision"),
        ("val_sensitivity", "train_sensitivity", "Sensitivity (Recall)"),
        ("val_pr_auc", None, "Val PR-AUC"),
    ]
    for ax, (vk, tk, title) in zip(axes.flat, panels):
        ax.plot(epochs, metrics[vk], "o-", label=f"Val {title}")
        if tk:
            ax.plot(epochs, metrics[tk], "o-", label=f"Train {title}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Decoupled Fast — Validation Metrics", y=1.02)
    fig.tight_layout()
    p = output_dir / "metric_curves.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    return saved


@torch.inference_mode()
def eval_decoupled_checkpoint(
    ckpt_path: Path,
    val_csv: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    net = build_3d_architecture(model_name="resnet3d", pooling="topk", topk_ratio=0.5)
    hparams.pop("net", None)
    model = DecoupledModelModule(net=net, **hparams)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if ckpt.get("epoch", 0) >= hparams.get("stage2_start_epoch", 11):
        model._stage2_active = True
        model.logit_adjuster.enable()
    model.eval()
    model.float()
    model.to(device)

    dataset = CachedVideoDataset(
        csv_path=str(val_csv),
        size=(96, 128, 128),
        cache_dir=str(cache_dir),
        strict_cache=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.float().to(device)
        if batch_x.ndim == 4:
            batch_x = batch_x.unsqueeze(1)
        logits = model(batch_x)
        logits, batch_y = model.prepare_for_bce_loss(logits, batch_y.to(device).float())
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs_list.append(probs)
        labels_list.append(batch_y.int().cpu().numpy().reshape(-1))

    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return y_true, y_prob, y_pred


@torch.inference_mode()
def eval_resnet3d_checkpoint(
    ckpt_path: Path,
    val_csv: Path,
    data_root: Path,
    batch_size: int,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from erdes.data.components.erdes_dataset import VideoDataset

    model = ModelModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.float()
    model.to(device)

    dataset = VideoDataset(
        csv_path=str(val_csv),
        size=(96, 128, 128),
        data_root=str(data_root),
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
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {metrics['confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-log", type=Path, default=DEFAULT_FAST_LOG)
    parser.add_argument("--fast-run-dir", type=Path, default=DEFAULT_FAST_RUN)
    parser.add_argument("--fast-ckpt", type=Path, default=DEFAULT_FAST_CKPT)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/erdes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fast_plots = args.fast_run_dir / "plots"
    fast_plots.mkdir(parents=True, exist_ok=True)
    base_plots = args.base_run_dir / "plots"
    base_plots.mkdir(parents=True, exist_ok=True)

    print("Parsing decoupled fast training log ...")
    metrics = parse_decoupled_log(args.fast_log)
    print(f"Epochs parsed: {metrics['epochs']}")

    print("\n--- Per-epoch validation metrics (from log) ---")
    print(f"{'Ep':>3} {'Loss':>7} {'F1':>7} {'PR-AUC':>7} {'Prec':>7} {'Sens':>7}")
    for i, e in enumerate(metrics["epochs"]):
        print(
            f"{e:3d} {metrics['val_loss'][i]:7.3f} {metrics['val_f1'][i]:7.3f} "
            f"{metrics['val_pr_auc'][i]:7.3f} {metrics['val_precision'][i]:7.3f} "
            f"{metrics['val_sensitivity'][i]:7.3f}"
        )

    saved = plot_metric_curves(metrics, fast_plots)
    for p in saved:
        print(f"Saved: {p}")

    if not args.fast_ckpt.exists():
        args.fast_ckpt = args.fast_run_dir / "checkpoints/resnet3d_decoupled/last.ckpt"
    print(f"\nEvaluating fast checkpoint: {args.fast_ckpt}")
    y_true, y_prob, y_pred = eval_decoupled_checkpoint(
        args.fast_ckpt,
        args.val_csv,
        args.cache_dir,
        args.batch_size,
        args.device,
    )
    m05 = binary_metrics(y_true, y_pred, y_prob)
    print_metrics_table("Decoupled Fast (best ckpt, thr=0.5)", m05, 0.5)

    thr_res = find_best_threshold_under_precision_constraint(y_true, y_prob, min_precision=0.5)
    y_pred_opt = (y_prob >= thr_res["threshold"]).astype(np.int64)
    m_opt = binary_metrics(y_true, y_pred_opt, y_prob)
    print_metrics_table("Decoupled Fast (PR-optimized thr)", m_opt, thr_res["threshold"])

    plot_confusion(
        np.array(m05["confusion_matrix"]),
        fast_plots / "confusion_matrix_val_thr0.5.png",
        "Decoupled Fast — Confusion Matrix (thr=0.5)",
    )
    plot_pr_curve(y_true, y_prob, fast_plots / "pr_curve_val.png", thr_res["threshold"])
    print(f"Saved: {fast_plots / 'pr_curve_val.png'}")
    print(f"Saved: {fast_plots / 'confusion_matrix_val_thr0.5.png'}")

    with (fast_plots / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "per_epoch_log": metrics,
                "checkpoint_eval_thr_0.5": m05,
                "checkpoint_eval_pr_optimized": {**m_opt, "threshold": thr_res["threshold"]},
            },
            f,
            indent=2,
        )

    if not args.base_ckpt.exists():
        print(f"\nBaseline ckpt not found: {args.base_ckpt}")
        return

    print(f"\nEvaluating baseline resnet3d (15 ep): {args.base_ckpt}")
    y_true_b, y_prob_b, y_pred_b = eval_resnet3d_checkpoint(
        args.base_ckpt,
        args.val_csv,
        args.data_root,
        args.batch_size,
        args.device,
    )
    m_base = binary_metrics(y_true_b, y_pred_b, y_prob_b)
    print_metrics_table("Original ResNet3D 15ep (thr=0.5)", m_base, 0.5)

    thr_base = find_best_threshold_under_precision_constraint(
        y_true_b, y_prob_b, min_precision=0.5
    )
    plot_pr_curve(
        y_true_b,
        y_prob_b,
        base_plots / "pr_curve_val.png",
        thr_base["threshold"],
    )
    print(f"Saved: {base_plots / 'pr_curve_val.png'}")
    plot_confusion(
        np.array(m_base["confusion_matrix"]),
        base_plots / "confusion_matrix_val.png",
        "Original ResNet3D 15ep — Confusion Matrix (thr=0.5)",
    )
    print(f"Saved: {base_plots / 'confusion_matrix_val.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    p1, r1, _ = precision_recall_curve(y_true, y_prob)
    p2, r2, _ = precision_recall_curve(y_true_b, y_prob_b)
    ax.plot(r1, p1, label=f"Decoupled Fast (AUC={auc(r1, p1):.3f})")
    ax.plot(r2, p2, label=f"ResNet3D 15ep (AUC={auc(r2, p2):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation PR Curves — Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    cmp_path = PROJECT_ROOT / "logs/plots/pr_curve_comparison.png"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cmp_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cmp_path}")


if __name__ == "__main__":
    main()
