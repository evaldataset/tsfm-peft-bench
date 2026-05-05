from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_head_only(
    model: nn.Module,
    head_module_patterns: list[str] | None = None,
) -> nn.Module:
    """Enable training only for head-like modules.

    Args:
        model: Original PyTorch model.
        head_module_patterns: Name patterns used to match trainable head params.

    Returns:
        The same model with in-place requires_grad updates.
    """

    if head_module_patterns is None:
        head_module_patterns = ["head", "lm_head", "output", "classifier", "forecast"]

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in head_module_patterns):
            param.requires_grad = True
            unfrozen_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    ratio = (unfrozen_count / total_params * 100) if total_params > 0 else 0.0

    if unfrozen_count == 0:
        logger.warning(
            "Head-only: no parameters matched patterns=%s",
            head_module_patterns,
        )
    else:
        logger.info(
            "Head-only applied: trainable=%s (%.2f%%)",
            f"{unfrozen_count:,}",
            ratio,
        )

    return model
