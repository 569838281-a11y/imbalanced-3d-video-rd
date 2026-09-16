"""Validation-set PR curve analysis and threshold moving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve


@torch.inference_mode()
def collect_binary_probs_and_labels(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    logit_adjuster=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on dataloader; return (y_true, y_prob) as numpy arrays."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        # Cached tensors may be float16; keep compute in float32 for BCE/sigmoid stability.
        if batch_x.dtype != torch.float32:
            batch_x = batch_x.float()
        logits = model(batch_x)
        if logit_adjuster is not None:
            logits = logit_adjuster(logits)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        labels = batch_y.detach().cpu().numpy().astype(np.int64)
        all_probs.append(np.asarray(probs).reshape(-1))
        all_labels.append(np.asarray(labels).reshape(-1))

    return np.concatenate(all_labels), np.concatenate(all_probs)


def find_best_threshold_under_precision_constraint(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.5,
    default_threshold: float = 0.5,
) -> dict:
    """
    On the PR curve, pick threshold with max recall subject to precision >= min_precision.
    Returns dict with threshold, recall, precision, f1, pr_auc.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))

    best_threshold = default_threshold
    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= min_precision and r >= best_recall:
            best_recall = float(r)
            best_precision = float(p)
            best_threshold = float(t)
            if p + r > 0:
                best_f1 = float(2 * p * r / (p + r))

    if best_recall == 0.0:
        preds = (y_prob >= default_threshold).astype(np.int64)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        best_precision = tp / (tp + fp + 1e-8)
        best_recall = tp / (tp + fn + 1e-8)
        best_f1 = (
            2 * best_precision * best_recall / (best_precision + best_recall + 1e-8)
        )
        best_threshold = default_threshold

    preds = (y_prob >= best_threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    return {
        "threshold": best_threshold,
        "recall": best_recall,
        "sensitivity": best_recall,
        "precision": best_precision,
        "f1": best_f1,
        "pr_auc": pr_auc,
        "min_precision_constraint": min_precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_positive": int(y_true.sum()),
        "support_negative": int((y_true == 0).sum()),
    }


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Union[str, Path],
    best_threshold: Optional[float] = None,
) -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, label=f"PR curve (AUC={pr_auc:.3f})")
    if best_threshold is not None:
        preds = (y_prob >= best_threshold).astype(np.int64)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        op = tp / (tp + fp + 1e-8)
        ore = tp / (tp + fn + 1e-8)
        ax.scatter([ore], [op], c="red", s=40, zorder=5, label=f"thr={best_threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_threshold_results(results: dict, output_path: Union[str, Path]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
