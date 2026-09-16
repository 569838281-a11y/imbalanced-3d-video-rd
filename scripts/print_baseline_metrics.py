"""Print baseline resnet3d 15-epoch validation metrics from log and best ckpt."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG = PROJECT_ROOT / "logs/latest_train_output.log"
CKPT = (
    PROJECT_ROOT
    / "logs/train/runs/rd/resnet3d/2026-05-29_22-02-10/checkpoints/resnet3d/resnet3d_best_epoch_015.ckpt"
)


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_log() -> dict[int, dict[str, float]]:
    text = re.sub(r"\s+", " ", read_log_text(LOG))
    pat = re.compile(
        r"Epoch (\d+): Val Loss: ([0-9.]+), Val Acc: ([0-9.]+), "
        r"Val Precision: ([0-9.]+), Val Sensitivity: ([0-9.]+), "
        r"Val Specificity: ([0-9.]+), Val F1: ([0-9.]+)"
    )
    rows: dict[int, dict[str, float]] = {}
    for m in pat.finditer(text):
        e = int(m.group(1))
        acc = float(m.group(3))
        if e == 0 and acc == 0.0:
            continue
        rows[e] = {
            "accuracy": acc,
            "precision": float(m.group(4)),
            "sensitivity": float(m.group(5)),
            "specificity": float(m.group(6)),
            "f1": float(m.group(7)),
        }
    return rows


def main() -> None:
    rows = parse_log()
    print("=== 基础 ResNet3D 15 轮训练 — 各 Epoch 验证集指标（日志） ===\n")
    print(f"{'Epoch':>5} {'Accuracy':>10} {'Precision':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8}")
    print("-" * 62)
    for e in sorted(rows):
        r = rows[e]
        print(
            f"{e:5d} {r['accuracy']:10.4f} {r['precision']:10.4f} "
            f"{r['sensitivity']:12.4f} {r['specificity']:12.4f} {r['f1']:8.4f}"
        )

    if CKPT.exists():
        import torch

        from scripts.plot_decoupled_fast_results import binary_metrics, eval_resnet3d_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== 基础 ResNet3D best checkpoint (epoch 15) — 验证集重算 thr=0.5 ===\n")
        y_true, y_prob, y_pred = eval_resnet3d_checkpoint(
            CKPT,
            PROJECT_ROOT / "data/splits/non_rd_vs_rd/val.csv",
            PROJECT_ROOT / "data/erdes",
            batch_size=4,
            device=device,
        )
        m = binary_metrics(y_true, y_pred, y_prob)
        for k in ("accuracy", "precision", "sensitivity", "specificity", "f1"):
            labels = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "sensitivity": "Sensitivity",
                "specificity": "Specificity",
                "f1": "F1",
            }
            print(f"  {labels[k]:12s}: {m[k]:.4f}")
    else:
        print(f"\n(Checkpoint not found: {CKPT})")


if __name__ == "__main__":
    main()
