"""
Two-stage decoupled training LightningModule for imbalanced RD vs non-RD classification.

Stage 1: full-network representation learning + optional Mixup.
Stage 2: frozen backbone, trainable head + Logit Adjustment.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.classification.accuracy import Accuracy

from erdes.training.backbone_utils import freeze_backbone
from erdes.training.class_priors import ClassPriorEstimator
from erdes.training.gpu_video_ops import gpu_preprocess_batch
from erdes.training.logit_adjustment import LogitAdjuster
from erdes.training.mixup import mixup_data


def _confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, int, int]:
    preds = preds.view(-1).int()
    targets = targets.view(-1).int()
    tp = int(((preds == 1) & (targets == 1)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    return tp, fp, fn, tn


def _metrics_at_threshold(probs: torch.Tensor, targets: torch.Tensor, thr: float) -> Dict[str, float]:
    preds = (probs >= thr).int()
    tp, fp, fn, tn = _confusion_counts(preds, targets)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "acc": acc,
    }


class DecoupledModelModule(LightningModule):
    # Thresholds for detailed recall / PR reporting each val epoch
    REPORT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        compile: bool,
        train_csv: str,
        stage1_end_epoch: int = 35,
        stage2_start_epoch: int = 35,
        total_epochs: int = 50,
        logit_adjustment_tau: float = 1.0,
        mixup_alpha: float = 0.2,
        mixup_enabled: bool = True,
        stage2_lr: Optional[float] = None,
        decision_threshold: float = 0.5,
        label_column: str = "label",
        gpu_preprocess: bool = False,
        video_size: Tuple[int, int, int] = (96, 128, 128),
        compute_train_metrics: bool = True,
        metrics_csv_name: str = "epoch_metrics.csv",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = nn.BCEWithLogitsLoss()

        prior_estimator = ClassPriorEstimator.from_csv(train_csv, label_column=label_column)
        self.logit_adjuster = LogitAdjuster(
            prior_estimator=prior_estimator,
            tau=logit_adjustment_tau,
            enabled=False,
        )

        self.stage1_end_epoch = stage1_end_epoch
        self.stage2_start_epoch = stage2_start_epoch
        self.total_epochs = total_epochs
        self.mixup_alpha = mixup_alpha
        self.mixup_enabled = mixup_enabled
        self.stage2_lr = stage2_lr
        self.decision_threshold = decision_threshold
        self.gpu_preprocess = gpu_preprocess
        self.video_size = tuple(int(x) for x in video_size)
        self._stage2_active = False
        self._optimizer_reconfigured = False
        self._val_probs: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._metrics_csv_name = metrics_csv_name

        self._init_metrics()

    def _init_metrics(self) -> None:
        thr = self.decision_threshold
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = Accuracy(task="binary", threshold=thr)
        self.val_acc = Accuracy(task="binary", threshold=thr)
        self.test_acc = Accuracy(task="binary", threshold=thr)

        self.train_precision = BinaryPrecision(threshold=thr)
        self.train_sensitivity = BinaryRecall(threshold=thr)
        self.train_specificity = BinarySpecificity(threshold=thr)
        self.train_f1 = BinaryF1Score(threshold=thr)

        self.val_precision = BinaryPrecision(threshold=thr)
        self.val_sensitivity = BinaryRecall(threshold=thr)
        self.val_specificity = BinarySpecificity(threshold=thr)
        self.val_f1 = BinaryF1Score(threshold=thr)
        self.val_pr_auc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

        self.test_precision = BinaryPrecision(threshold=thr)
        self.test_sensitivity = BinaryRecall(threshold=thr)
        self.test_specificity = BinarySpecificity(threshold=thr)
        self.test_f1 = BinaryF1Score(threshold=thr)

        self.val_f1_best = MaxMetric()
        self.val_pr_auc_best = MaxMetric()
        self.val_recall_best = MaxMetric()

    def _update_metric_thresholds(self, threshold: float) -> None:
        self.decision_threshold = threshold
        for metric in (
            self.train_acc,
            self.val_acc,
            self.test_acc,
            self.train_precision,
            self.train_sensitivity,
            self.train_specificity,
            self.train_f1,
            self.val_precision,
            self.val_sensitivity,
            self.val_specificity,
            self.val_f1,
            self.test_precision,
            self.test_sensitivity,
            self.test_specificity,
            self.test_f1,
        ):
            if hasattr(metric, "threshold"):
                metric.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return self.logit_adjuster(logits)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        x, y = batch
        if x.dtype == torch.float16:
            x = x.float()
        if not self.gpu_preprocess:
            return x, y
        target = self.video_size
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.shape[2:] == target and x.max() <= 1.5:
            return x.float(), y
        x = gpu_preprocess_batch(x, target, normalize=True)
        return x, y

    def on_train_start(self) -> None:
        self.print(
            f"[Decoupled] Class priors: {ClassPriorEstimator.from_csv(self.hparams.train_csv)}"
        )
        self.print(
            f"[Decoupled] Stage1 epochs 1-{self.stage1_end_epoch}, "
            f"Stage2 epochs {self.stage2_start_epoch + 1}-{self.total_epochs}"
        )
        self.print(
            f"[Decoupled] Decision threshold={self.decision_threshold:.3f}; "
            f"detailed val recall will be reported at thr={list(self.REPORT_THRESHOLDS)}"
        )

    def on_train_epoch_start(self) -> None:
        # Use >= so resume from mid/late Stage-2 still freezes backbone + enables LA.
        if self.current_epoch >= self.stage2_start_epoch and not self._stage2_active:
            self._enter_stage2()

        self.train_loss.reset()
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_sensitivity.reset()
        self.train_specificity.reset()
        self.train_f1.reset()

    def _enter_stage2(self) -> None:
        self._stage2_active = True
        head_names = freeze_backbone(self.net)
        self.logit_adjuster.enable()
        self.print(
            f"[Decoupled] >>> Entering Stage 2 at epoch {self.current_epoch + 1}: "
            f"backbone frozen, Logit Adjustment ON, trainable head params: {len(head_names)}"
        )
        self._reconfigure_optimizer_for_stage2()

    def _reconfigure_optimizer_for_stage2(self) -> None:
        if self._optimizer_reconfigured or self.trainer is None:
            return

        trainable = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Stage 2 has no trainable parameters after freezing backbone.")

        lr = self.stage2_lr
        weight_decay = 0.0
        if lr is None:
            opt_partial = self.hparams.optimizer
            if hasattr(opt_partial, "keywords"):
                lr = opt_partial.keywords.get("lr", 1.5e-5)
                weight_decay = opt_partial.keywords.get("weight_decay", 0.0)
            else:
                lr = 1.5e-5

        new_optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

        self.trainer.optimizers = [new_optimizer]

        if self.hparams.scheduler is not None and self.trainer.lr_scheduler_configs:
            new_scheduler = self.hparams.scheduler(optimizer=new_optimizer)
            self.trainer.lr_scheduler_configs[0].scheduler = new_scheduler

        self._optimizer_reconfigured = True
        self.print(f"[Decoupled] Optimizer rebuilt for Stage 2 (lr={lr}, n_params={len(trainable)})")

    def _apply_mixup_if_stage1(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        if self._stage2_active or not self.mixup_enabled or self.mixup_alpha <= 0:
            return x, y
        return mixup_data(x, y, alpha=self.mixup_alpha)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self._apply_mixup_if_stage1(batch)
        logits = self.forward(x)
        logits, y = self.prepare_for_bce_loss(logits, y)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= self.decision_threshold).int().view(-1)
        return loss, preds, y

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        if self.hparams.compute_train_metrics:
            t = targets.int()
            self.train_acc(preds, t)
            self.train_precision(preds, t)
            self.train_sensitivity(preds, t)
            self.train_specificity(preds, t)
            self.train_f1(preds, t)

        stage = 2 if self._stage2_active else 1
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.compute_train_metrics:
            self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "train/sensitivity",
                self.train_sensitivity,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("train/recall", self.train_sensitivity, on_step=False, on_epoch=True)
            self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
            self.log("train/specificity", self.train_specificity, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/stage", float(stage), on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        stage = 2 if self._stage2_active else 1
        if self.hparams.compute_train_metrics:
            sens = float(self.train_sensitivity.compute())
            self.print(
                f"Epoch {self.current_epoch} [Stage {stage}]: "
                f"Train Loss: {self.train_loss.compute():.4f}, "
                f"Train Acc: {self.train_acc.compute():.4f}, "
                f"Train F1: {self.train_f1.compute():.4f}, "
                f"Train Recall/Sensitivity: {sens:.4f}, "
                f"Train Precision: {self.train_precision.compute():.4f}, "
                f"Train Specificity: {self.train_specificity.compute():.4f}"
            )
        else:
            self.print(
                f"Epoch {self.current_epoch} [Stage {stage}]: "
                f"Train Loss: {self.train_loss.compute():.4f} (train metrics disabled)"
            )

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_sensitivity.reset()
        self.val_specificity.reset()
        self.val_f1.reset()
        self.val_pr_auc.reset()
        self.val_auroc.reset()
        self._val_probs = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self.forward(x)
        logits, y = self.prepare_for_bce_loss(logits, y.float())
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= self.decision_threshold).int().view(-1)
        targets = y.int().view(-1)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_sensitivity(preds, targets)
        self.val_specificity(preds, targets)
        self.val_f1(preds, targets)
        self.val_pr_auc(probs, targets)
        self.val_auroc(probs, targets)

        self._val_probs.append(probs.detach().float().cpu())
        self._val_targets.append(targets.detach().cpu())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sensitivity", self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/specificity", self.val_specificity, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/pr_auc", self.val_pr_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        f1 = float(self.val_f1.compute())
        pr_auc = float(self.val_pr_auc.compute())
        sens = float(self.val_sensitivity.compute())
        prec = float(self.val_precision.compute())
        spec = float(self.val_specificity.compute())
        acc = float(self.val_acc.compute())
        auroc = float(self.val_auroc.compute())

        self.val_f1_best(f1)
        self.val_pr_auc_best(pr_auc)
        self.val_recall_best(sens)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/pr_auc_best", self.val_pr_auc_best.compute(), sync_dist=True)
        self.log("val/recall_best", self.val_recall_best.compute(), sync_dist=True, prog_bar=True)

        stage = 2 if self._stage2_active else 1
        thr0 = self.decision_threshold

        row: Dict[str, Any] = {
            "epoch": int(self.current_epoch),
            "stage": stage,
            "decision_threshold": thr0,
            "val_loss": float(self.val_loss.compute()),
            "val_acc": acc,
            "val_f1": f1,
            "val_precision": prec,
            "val_recall": sens,
            "val_sensitivity": sens,
            "val_specificity": spec,
            "val_pr_auc": pr_auc,
            "val_auroc": auroc,
            "val_f1_best": float(self.val_f1_best.compute()),
            "val_pr_auc_best": float(self.val_pr_auc_best.compute()),
            "val_recall_best": float(self.val_recall_best.compute()),
        }

        if self._val_probs and self._val_targets:
            probs = torch.cat(self._val_probs, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            cm = _metrics_at_threshold(probs, targets, thr0)
            row.update(
                {
                    "tp": cm["tp"],
                    "fp": cm["fp"],
                    "fn": cm["fn"],
                    "tn": cm["tn"],
                    "n_pos": int(targets.sum().item()),
                    "n_neg": int((targets == 0).sum().item()),
                }
            )
            self.log("val/tp", float(cm["tp"]), sync_dist=True)
            self.log("val/fp", float(cm["fp"]), sync_dist=True)
            self.log("val/fn", float(cm["fn"]), sync_dist=True)
            self.log("val/tn", float(cm["tn"]), sync_dist=True)

            self.print("=" * 72)
            self.print(f"Epoch {self.current_epoch} [Stage {stage}] VAL @ thr={thr0:.3f}")
            self.print(
                f"  Acc={acc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  "
                f"Recall/Sensitivity={sens:.4f}  Specificity={spec:.4f}"
            )
            self.print(
                f"  PR-AUC={pr_auc:.4f}  ROC-AUC={auroc:.4f}  "
                f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}  "
                f"(pos={row['n_pos']} neg={row['n_neg']})"
            )
            self.print("  --- Recall at multiple thresholds ---")
            for t in self.REPORT_THRESHOLDS:
                m = _metrics_at_threshold(probs, targets, t)
                row[f"recall_thr{t:.1f}"] = m["recall"]
                row[f"precision_thr{t:.1f}"] = m["precision"]
                row[f"f1_thr{t:.1f}"] = m["f1"]
                self.log(f"val/recall_thr_{t:.1f}", m["recall"], sync_dist=True)
                self.log(f"val/precision_thr_{t:.1f}", m["precision"], sync_dist=True)
                self.log(f"val/f1_thr_{t:.1f}", m["f1"], sync_dist=True)
                self.print(
                    f"  thr={t:.1f}: Recall={m['recall']:.4f}  "
                    f"Precision={m['precision']:.4f}  F1={m['f1']:.4f}  "
                    f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
                )
            self.print("=" * 72)
        else:
            self.print(
                f"Epoch {self.current_epoch} [Stage {stage}]: "
                f"Val Loss: {row['val_loss']:.4f}, F1={f1:.4f}, "
                f"Recall={sens:.4f}, Precision={prec:.4f}, thr={thr0:.3f}"
            )

        self._append_metrics_csv(row)

    def _append_metrics_csv(self, row: Dict[str, Any]) -> None:
        clean = {k: v for k, v in row.items() if " " not in str(k)}
        try:
            out_dir = Path(self.trainer.default_root_dir) if self.trainer else Path(".")
            path = out_dir / self._metrics_csv_name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.is_file()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(clean.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(clean)
        except Exception as exc:
            self.print(f"[Decoupled] Warning: could not write metrics CSV: {exc}")

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self.forward(x)
        logits, y = self.prepare_for_bce_loss(logits, y.float())
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= self.decision_threshold).int().view(-1)
        targets = y.int().view(-1)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_sensitivity(preds, targets)
        self.test_specificity(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/sensitivity", self.test_sensitivity, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_sensitivity, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/specificity", self.test_specificity, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def prepare_for_bce_loss(self, preds, targets):
        targets = targets.float()
        if preds.ndim > 1 and preds.shape[-1] == 1:
            preds = preds.view(-1)
        if targets.ndim > 1 and targets.shape[-1] == 1:
            targets = targets.view(-1)
        if preds.shape != targets.shape:
            preds = preds.view(-1)
            targets = targets.view(-1)
        return preds, targets

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def set_decision_threshold(self, threshold: float) -> None:
        self._update_metric_thresholds(threshold)
        self.print(f"[Decoupled] Decision threshold updated to {threshold:.4f}")
