"""Logit Adjustment (Menon et al.) for long-tailed binary classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from erdes.training.class_priors import ClassPriorEstimator


class LogitAdjuster(nn.Module):
    """
    Adjusts a single positive-class logit:
        z' = z - tau * (log(pi_1) - log(pi_0))

    Shifts the decision boundary to improve minority-class recall at inference.
    """

    def __init__(
        self,
        prior_estimator: ClassPriorEstimator,
        tau: float = 1.0,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        priors = prior_estimator.as_tensor
        log_priors = torch.log(priors.clamp(min=1e-8))
        self.register_buffer("log_pi_neg", log_priors[0])
        self.register_buffer("log_pi_pos", log_priors[1])
        self.tau = tau
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return logits
        offset = self.tau * (self.log_pi_pos - self.log_pi_neg)
        return logits - offset

    def extra_repr(self) -> str:
        return f"tau={self.tau}, enabled={self._enabled}"
