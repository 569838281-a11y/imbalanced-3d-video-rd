"""Post-training PR-curve threshold search on the validation set."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only

from erdes.models.decoupled_model_module import DecoupledModelModule
from erdes.training.threshold_search import (
    collect_binary_probs_and_labels,
    find_best_threshold_under_precision_constraint,
    plot_pr_curve,
    save_threshold_results,
)


class PRThresholdSearchCallback(Callback):
    """
    After fit() completes (epoch 50), sweep validation PR curve and store
    the best threshold under a minimum precision constraint.
    """

    def __init__(
        self,
        min_precision: float = 0.5,
        save_plot: bool = True,
        results_filename: str = "best_threshold.json",
        plot_filename: str = "pr_curve_val.png",
    ) -> None:
        super().__init__()
        self.min_precision = min_precision
        self.save_plot = save_plot
        self.results_filename = results_filename
        self.plot_filename = plot_filename

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not isinstance(pl_module, DecoupledModelModule):
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        datamodule.setup("validate")
        val_loader = datamodule.val_dataloader()
        device = pl_module.device

        y_true, y_prob = collect_binary_probs_and_labels(
            model=pl_module,
            dataloader=val_loader,
            device=device,
        )

        results = find_best_threshold_under_precision_constraint(
            y_true=y_true,
            y_prob=y_prob,
            min_precision=self.min_precision,
        )

        output_dir = Path(trainer.default_root_dir)
        save_threshold_results(results, output_dir / self.results_filename)

        pl_module.set_decision_threshold(results["threshold"])

        if self.save_plot:
            plot_pr_curve(
                y_true,
                y_prob,
                output_dir / self.plot_filename,
                best_threshold=results["threshold"],
            )

        pl_module.print(
            "[PR Threshold Search] "
            f"best_thr={results['threshold']:.4f}, "
            f"recall/sensitivity={results['recall']:.4f}, "
            f"precision={results['precision']:.4f}, "
            f"f1={results['f1']:.4f}, "
            f"pr_auc={results['pr_auc']:.4f}, "
            f"TP={results.get('tp')} FP={results.get('fp')} "
            f"FN={results.get('fn')} TN={results.get('tn')} "
            f"(precision>={self.min_precision})"
        )
