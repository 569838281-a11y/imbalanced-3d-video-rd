"""Light Mixup augmentation for stage-1 representation learning."""

from __future__ import annotations

import torch


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Beta(alpha, alpha) mixup for binary video batches.
    Targets are mixed as soft labels in [0, 1] for BCEWithLogitsLoss.
    """
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y.float()

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    mixed_y = lam * y.float() + (1.0 - lam) * y[index].float()
    return mixed_x, mixed_y
