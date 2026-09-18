"""SSL-protect fine-tune: freeze backbone first, then unfreeze for full fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import F1Score, Precision, Recall, Specificity
from torchmetrics.classification.accuracy import Accuracy

from erdes.training.backbone_utils import freeze_backbone, unfreeze_all


class SSLProtectFinetuneModule(LightningModule):
    """
    Stage A (epochs 0 .. freeze_epochs-1): SSL/backbone frozen, train head only.
    Stage B (freeze_epochs ..): unfreeze all, continue with finetune_lr.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        compile: bool = False,
        freeze_epochs: int = 15,
        finetune_lr: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.freeze_epochs = int(freeze_epochs)
        self.finetune_lr = float(finetune_lr)
        self._unfrozen = False

        self.train_acc = Accuracy(task="binary", threshold=0.5)
        self.val_acc = Accuracy(task="binary", threshold=0.5)
        self.test_acc = Accuracy(task="binary", threshold=0.5)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.train_precision = Precision(task="binary", threshold=0.5)
        self.train_sensitivity = Recall(task="binary", threshold=0.5)
        self.train_specificity = Specificity(task="binary", threshold=0.5)
        self.train_f1 = F1Score(task="binary", threshold=0.5)
        self.val_precision = Precision(task="binary", threshold=0.5)
        self.val_sensitivity = Recall(task="binary", threshold=0.5)
        self.val_specificity = Specificity(task="binary", threshold=0.5)
        self.val_f1 = F1Score(task="binary", threshold=0.5)
        self.test_precision = Precision(task="binary", threshold=0.5)
        self.test_sensitivity = Recall(task="binary", threshold=0.5)
        self.test_specificity = Specificity(task="binary", threshold=0.5)
        self.test_f1 = F1Score(task="binary", threshold=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def prepare_for_bce_loss(self, preds, targets):
        targets = targets.float()
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.view(-1)
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.view(-1)
        if preds.shape != targets.shape:
            preds = preds.view(-1)
            targets = targets.view(-1)
        return preds, targets

    def model_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        logits, y = self.prepare_for_bce_loss(logits, y)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).int()
        return loss, preds, y.int()

    def on_fit_start(self) -> None:
        head = freeze_backbone(self.net)
        self.print(
            f"[SSL-Protect] Stage A: backbone frozen for first {self.freeze_epochs} epochs; "
            f"trainable head tensors={len(head)}"
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch >= self.freeze_epochs and not self._unfrozen:
            self._enter_finetune()
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_sensitivity.reset()
        self.train_specificity.reset()
        self.train_f1.reset()
        stage = "B-finetune" if self._unfrozen else "A-freeze"
        self.print(f"[SSL-Protect] Epoch {self.current_epoch} stage={stage}")

    def _enter_finetune(self) -> None:
        self._unfrozen = True
        n = unfreeze_all(self.net)
        self.print(
            f"[SSL-Protect] >>> Stage B at epoch {self.current_epoch}: "
            f"unfreeze all ({n} tensors), lr -> {self.finetune_lr}"
        )
        if self.trainer is None:
            return
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        new_opt = torch.optim.Adam(trainable, lr=self.finetune_lr, weight_decay=0.0)
        self.trainer.optimizers = [new_opt]
        if self.hparams.scheduler is not None and self.trainer.lr_scheduler_configs:
            new_sched = self.hparams.scheduler(optimizer=new_opt)
            self.trainer.lr_scheduler_configs[0].scheduler = new_sched

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_sensitivity(preds, targets)
        self.train_specificity(preds, targets)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.print(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {self.train_loss.compute():.3f}, "
            f"Train Acc: {self.train_acc.compute():.3f}, "
            f"Train F1: {self.train_f1.compute():.3f}, "
        )

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_sensitivity.reset()
        self.val_specificity.reset()
        self.val_f1.reset()

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_sensitivity(preds, targets)
        self.val_specificity(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/sensitivity", self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/specificity", self.val_specificity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        # Logged only in Stage B; StageB* callbacks ignore Stage A entirely.
        if self._unfrozen:
            self.log(
                "val/f1_finetune",
                self.val_f1.compute(),
                sync_dist=True,
                prog_bar=True,
            )
        stage = "B" if self._unfrozen else "A"
        self.print(
            f"Epoch {self.current_epoch} [Stage {stage}]: "
            f"Val Loss: {self.val_loss.compute():.3f}, "
            f"Val Acc: {self.val_acc.compute():.3f}, "
            f"Val F1: {self.val_f1.compute():.3f}, "
        )

    def test_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_sensitivity(preds, targets)
        self.test_specificity(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Resume mid-run: mark Stage B if we already passed freeze_epochs.
        epoch = int(checkpoint.get("epoch", 0))
        if epoch >= self.freeze_epochs:
            self._unfrozen = True

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Start with head-only params (backbone already frozen in on_fit_start;
        # configure_optimizers may run before on_fit_start, so freeze here too).
        freeze_backbone(self.net)
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=trainable)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
