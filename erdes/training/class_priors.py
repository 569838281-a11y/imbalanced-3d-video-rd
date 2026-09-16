"""Estimate class priors from training CSV for logit adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch


@dataclass
class ClassPriorEstimator:
    """Stores binary class priors pi_0, pi_1 and log-priors for LA."""

    pi_negative: float
    pi_positive: float

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        label_column: str = "label",
        positive_label: int = 1,
    ) -> "ClassPriorEstimator":
        pi_neg, pi_pos = compute_binary_priors_from_csv(
            csv_path, label_column=label_column, positive_label=positive_label
        )
        return cls(pi_negative=pi_neg, pi_positive=pi_pos)

    @property
    def as_tensor(self) -> torch.Tensor:
        """Tensor [pi_0, pi_1] on CPU."""
        return torch.tensor([self.pi_negative, self.pi_positive], dtype=torch.float32)

    def to_dict(self) -> Dict[str, float]:
        return {"pi_negative": self.pi_negative, "pi_positive": self.pi_positive}

    def __repr__(self) -> str:
        return (
            f"ClassPriorEstimator(pi_negative={self.pi_negative:.4f}, "
            f"pi_positive={self.pi_positive:.4f})"
        )


def compute_binary_priors_from_csv(
    csv_path: Union[str, Path],
    label_column: str = "label",
    positive_label: int = 1,
) -> tuple[float, float]:
    """Return (pi_negative, pi_positive) from label counts."""
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise KeyError(f"Column '{label_column}' not in {csv_path}")

    labels = df[label_column].astype(int)
    n_pos = int((labels == positive_label).sum())
    n_neg = int((labels != positive_label).sum())
    total = n_pos + n_neg
    if total == 0:
        raise ValueError(f"No samples in {csv_path}")

    pi_pos = n_pos / total
    pi_neg = n_neg / total
    return pi_neg, pi_pos
