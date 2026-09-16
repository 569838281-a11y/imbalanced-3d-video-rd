"""Flush training progress to stdout (works with Tee-Object / log files)."""

from __future__ import annotations

import sys
from pathlib import Path

from lightning import Callback, Trainer
from lightning.pytorch import LightningModule


class TrainingProgressCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 25,
        progress_log_file: str = "",
    ) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.progress_log_file = progress_log_file.strip()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        n = trainer.num_training_batches
        print(
            f"\n[Train] Epoch {trainer.current_epoch} started — {n} batches "
            f"(log every {self.log_every_n_steps} steps)\n",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        n_batches = trainer.num_training_batches or "?"
        loss_val = None
        if hasattr(pl_module, "train_loss"):
            try:
                loss_val = pl_module.train_loss.compute().item()
            except Exception:
                loss_val = None

        stage = 2 if getattr(pl_module, "_stage2_active", False) else 1
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "n/a"
        msg = (
            f"[step {step}] epoch {trainer.current_epoch} "
            f"batch {batch_idx + 1}/{n_batches} stage={stage} train_loss(avg)={loss_str}"
        )
        print(msg, flush=True)
        sys.stdout.flush()
        if self.progress_log_file:
            Path(self.progress_log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
