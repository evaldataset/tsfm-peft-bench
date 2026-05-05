from __future__ import annotations

import logging
from importlib import import_module
from dataclasses import dataclass
from typing import Protocol, cast

import torch.nn as nn

logger = logging.getLogger(__name__)


class _TaskTypeEnum(Protocol):
    SEQ_2_SEQ_LM: object
    CAUSAL_LM: object


class _PrefixConfigFactory(Protocol):
    def __call__(
        self, *, num_virtual_tokens: int, task_type: object | None
    ) -> object: ...


class _GetPeftModel(Protocol):
    def __call__(self, model: nn.Module, peft_config: object) -> nn.Module: ...


class _PeftPrefixModule(Protocol):
    TaskType: _TaskTypeEnum
    PrefixTuningConfig: _PrefixConfigFactory
    get_peft_model: _GetPeftModel


@dataclass
class PrefixAdaptationConfig:
    """Prefix Tuning adaptation settings.

    Attributes:
        num_virtual_tokens: Number of virtual prompt tokens.
        task_type: HuggingFace PEFT TaskType string.
    """

    num_virtual_tokens: int = 32
    task_type: str = "SEQ_2_SEQ_LM"


def apply_prefix_tuning(model: nn.Module, config: PrefixAdaptationConfig) -> nn.Module:
    """Apply Prefix Tuning adaptation to a model.

    Args:
        model: Original PyTorch model.
        config: Prefix Tuning settings.

    Returns:
        PEFT model with Prefix Tuning.
    """

    peft_module = cast(_PeftPrefixModule, cast(object, import_module("peft")))
    task_type_enum = peft_module.TaskType

    task_type_mapping = {
        "SEQ_2_SEQ_LM": task_type_enum.SEQ_2_SEQ_LM,
        "CAUSAL_LM": task_type_enum.CAUSAL_LM,
    }
    task_type = task_type_mapping.get(config.task_type)

    prefix_config_cls = peft_module.PrefixTuningConfig
    get_peft_model = peft_module.get_peft_model

    prefix_config = prefix_config_cls(
        num_virtual_tokens=config.num_virtual_tokens,
        task_type=task_type,
    )

    peft_model = get_peft_model(model, prefix_config)

    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in peft_model.parameters())
    ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    logger.info(
        "Prefix Tuning applied: num_virtual_tokens=%d, trainable=%s (%.2f%%)",
        config.num_virtual_tokens,
        f"{trainable_params:,}",
        ratio,
    )

    return peft_model
