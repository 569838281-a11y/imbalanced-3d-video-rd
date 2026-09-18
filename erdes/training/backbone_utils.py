"""Freeze backbone / unfreeze classification head for decoupled stage 2."""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn

# Parameter name fragments that belong to the classification head.
HEAD_NAME_FRAGMENTS = (
    "cls.",
    ".cls.",
    "fc1",
    "fc2",
    "model.fc",
    "model.linear",
    "classifier",
)


def get_head_parameter_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, _ in model.named_parameters():
        if any(fragment in name for fragment in HEAD_NAME_FRAGMENTS):
            names.append(name)
    return names


def head_parameters(model: nn.Module) -> Iterable:
    head_names = set(get_head_parameter_names(model))
    for name, param in model.named_parameters():
        if name in head_names:
            yield param


def backbone_parameters(model: nn.Module) -> Iterable:
    head_names = set(get_head_parameter_names(model))
    for name, param in model.named_parameters():
        if name not in head_names:
            yield param


def freeze_backbone(model: nn.Module) -> List[str]:
    """Freeze non-head parameters; return list of head param names kept trainable."""
    head_names = get_head_parameter_names(model)
    if not head_names:
        raise RuntimeError(
            "No classification-head parameters found. "
            "Check HEAD_NAME_FRAGMENTS in backbone_utils.py for your architecture."
        )

    for name, param in model.named_parameters():
        param.requires_grad = name in head_names

    return head_names


def unfreeze_all(model: nn.Module) -> int:
    """Unfreeze every parameter; return count of trainable tensors."""
    n = 0
    for param in model.parameters():
        param.requires_grad = True
        n += 1
    return n
