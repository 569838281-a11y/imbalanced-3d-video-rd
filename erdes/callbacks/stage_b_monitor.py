"""Callbacks that only act after SSL-protect Stage B (backbone unfrozen)."""

from __future__ import annotations

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def _in_stage_b(pl_module) -> bool:
    return bool(getattr(pl_module, "_unfrozen", False))


class StageBModelCheckpoint(ModelCheckpoint):
    """Save best ckpt only after unfreeze; ignore Stage-A metrics."""

    def on_validation_end(self, trainer, pl_module) -> None:
        if not _in_stage_b(pl_module):
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not _in_stage_b(pl_module):
            return
        super().on_train_epoch_end(trainer, pl_module)


class StageBEarlyStopping(EarlyStopping):
    """Early-stop only after unfreeze; Stage-A does not consume patience."""

    def on_validation_end(self, trainer, pl_module) -> None:
        if not _in_stage_b(pl_module):
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not _in_stage_b(pl_module):
            return
        super().on_train_epoch_end(trainer, pl_module)
