"""Save encoder state_dict at end of each training epoch (lightweight vs full Lightning ckpt)."""

from __future__ import annotations

from pathlib import Path

import torch
from lightning import Callback, Trainer
from lightning.pytorch import LightningModule


class EncoderEpochCheckpointCallback(Callback):
    def __init__(
        self,
        dirpath: str,
        filename: str = "encoder_epoch_{epoch:03d}.pth",
        submodule: str = "backbone",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.submodule = submodule

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        self.dirpath.mkdir(parents=True, exist_ok=True)
        out = self.dirpath / self.filename.format(epoch=epoch)
        net = pl_module.net
        module = getattr(net, self.submodule, net)
        torch.save(module.state_dict(), out)
        print(f"Saved encoder weights -> {out}", flush=True)
