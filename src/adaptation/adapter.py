from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _TaskTypeEnum(Protocol):
    SEQ_2_SEQ_LM: object
    CAUSAL_LM: object
    FEATURE_EXTRACTION: object


class _IA3ConfigFactory(Protocol):
    def __call__(
        self,
        *,
        task_type: object | None,
        target_modules: list[str],
        feedforward_modules: list[str] | None,
        inference_mode: bool,
    ) -> object: ...


class _GetPeftModel(Protocol):
    def __call__(self, model: nn.Module, peft_config: object) -> nn.Module: ...


class _PeftIA3Module(Protocol):
    TaskType: _TaskTypeEnum
    IA3Config: _IA3ConfigFactory
    get_peft_model: _GetPeftModel


@dataclass
class AdapterAdaptationConfig:
    """Adapter adaptation settings.

    Attributes:
        bottleneck_size: Adapter bottleneck projection size.
        task_type: HuggingFace PEFT task type string.
    """

    bottleneck_size: int = 64
    task_type: str = "SEQ_2_SEQ_LM"


class _BottleneckAdapterLinear(nn.Module):
    """Linear layer with residual bottleneck adapter branch."""

    def __init__(self, base: nn.Linear, bottleneck_size: int) -> None:
        """Initialize adapter-wrapped linear module.

        Args:
            base: Frozen base linear layer.
            bottleneck_size: Adapter bottleneck width.
        """
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.down = nn.Linear(
            in_features=base.in_features,
            out_features=bottleneck_size,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.activation = nn.GELU()
        self.up = nn.Linear(
            in_features=bottleneck_size,
            out_features=base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual adapter branch.
        """
        return self.base(x) + self.up(self.activation(self.down(x)))


def _resolve_task_type(task_type_str: str) -> object | None:
    """Map string task type to PEFT TaskType.

    Args:
        task_type_str: TaskType string.

    Returns:
        TaskType instance or None.
    """
    peft_module = cast(_PeftIA3Module, cast(object, import_module("peft")))
    task_type_enum = peft_module.TaskType
    mapping = {
        "SEQ_2_SEQ_LM": task_type_enum.SEQ_2_SEQ_LM,
        "CAUSAL_LM": task_type_enum.CAUSAL_LM,
        "FEATURE_EXTRACTION": task_type_enum.FEATURE_EXTRACTION,
    }
    return mapping.get(task_type_str)


def _select_ia3_targets(model: nn.Module) -> tuple[list[str], list[str] | None]:
    """Select IA3 target module suffixes from an existing backbone.

    Args:
        model: Backbone model.

    Returns:
        Tuple of (target_modules, feedforward_modules).
    """
    attention_candidates = {"q", "k", "v", "o", "query", "key", "value"}
    feedforward_candidates = {"wi", "wo", "fc1", "fc2", "dense"}

    found_attention: set[str] = set()
    found_feedforward: set[str] = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if short_name in attention_candidates:
            found_attention.add(short_name)
        if short_name in feedforward_candidates:
            found_feedforward.add(short_name)

    target_modules = sorted(found_attention | found_feedforward)
    feedforward_modules = sorted(found_feedforward)
    if not feedforward_modules:
        return target_modules, None
    return target_modules, feedforward_modules


def _set_module(root: nn.Module, module_path: str, module: nn.Module) -> None:
    """Set a child module by dotted path.

    Args:
        root: Root model module.
        module_path: Dotted module path.
        module: Replacement module.
    """
    parts = module_path.split(".")
    if len(parts) == 1:
        setattr(root, parts[0], module)
        return

    parent_path = ".".join(parts[:-1])
    parent = root.get_submodule(parent_path)
    setattr(parent, parts[-1], module)


def _apply_bottleneck_fallback(
    model: nn.Module,
    config: AdapterAdaptationConfig,
) -> nn.Module:
    """Apply manual bottleneck adapters when IA3 is unavailable.

    Args:
        model: Backbone model.
        config: Adapter settings.

    Returns:
        Model with trainable bottleneck adapters.
    """
    for param in model.parameters():
        param.requires_grad = False

    target_candidates = {
        "q",
        "k",
        "v",
        "o",
        "wi",
        "wo",
        "query",
        "key",
        "value",
        "fc1",
        "fc2",
        "dense",
    }

    linear_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if short_name in target_candidates:
            linear_names.append(name)

    if not linear_names:
        linear_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]

    replaced = 0
    for name in linear_names:
        module = cast(nn.Linear, model.get_submodule(name))
        wrapped = _BottleneckAdapterLinear(module, config.bottleneck_size)
        _set_module(model, name, wrapped)
        replaced += 1

    logger.info(
        "Manual bottleneck adapters applied to %d linear modules (bottleneck=%d)",
        replaced,
        config.bottleneck_size,
    )
    return model


def _count_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Count trainable and total parameters.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (trainable_params, total_params, trainable_ratio_percent).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    return trainable_params, total_params, ratio


def apply_adapter(model: nn.Module, config: AdapterAdaptationConfig) -> nn.Module:
    """IA3 어댑터를 모델에 적용.

    PEFT IA3를 사용하며, IA3를 사용할 수 없는 경우 명시적 에러를 발생시킨다.
    결과 레이블의 오염을 방지하기 위해 silent fallback을 하지 않는다.

    Args:
        model: 원본 PyTorch 모델.
        config: 어댑터 적응 설정.

    Returns:
        IA3가 적용된 모델.

    Raises:
        RuntimeError: IA3를 적용할 수 없을 때.
    """
    try:
        peft_module = cast(_PeftIA3Module, cast(object, import_module("peft")))
    except ImportError:
        raise RuntimeError(
            "PEFT 패키지가 설치되지 않았습니다. "
            "`pip install peft` 후 다시 시도하세요. "
            "Bottleneck fallback은 결과 레이블 오염 방지를 위해 비활성화되었습니다."
        )

    if not hasattr(peft_module, "IA3Config"):
        raise RuntimeError(
            "설치된 PEFT 버전에 IA3Config가 없습니다. "
            "`pip install --upgrade peft` 후 다시 시도하세요."
        )

    target_modules, feedforward_modules = _select_ia3_targets(model)
    if not target_modules:
        raise RuntimeError(
            f"모델에서 IA3 호환 타겟 모듈을 찾을 수 없습니다: "
            f"{type(model).__name__}. "
            f"모델 아키텍처가 IA3를 지원하는지 확인하세요."
        )

    try:
        ia3_config_cls = peft_module.IA3Config
        get_peft_model = peft_module.get_peft_model
        task_type = _resolve_task_type(config.task_type)
        ia3_config = ia3_config_cls(
            task_type=task_type,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
            inference_mode=False,
        )
        adapted_model = get_peft_model(model, ia3_config)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"PEFT IA3 설정 실패: {exc}. "
            f"Bottleneck fallback은 결과 레이블 오염 방지를 위해 비활성화되었습니다."
        ) from exc

    trainable_params, _, ratio = _count_parameters(adapted_model)
    logger.info(
        "Adapter applied (IA3): task_type=%s, trainable=%s (%.2f%%)",
        config.task_type,
        f"{trainable_params:,}",
        ratio,
    )
    return adapted_model
