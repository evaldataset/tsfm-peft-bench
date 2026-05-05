from __future__ import annotations

import logging
from importlib import import_module
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

import torch.nn as nn

logger = logging.getLogger(__name__)


class _TaskTypeEnum(Protocol):
    SEQ_2_SEQ_LM: object
    CAUSAL_LM: object
    FEATURE_EXTRACTION: object


class _LoraConfigFactory(Protocol):
    def __call__(
        self,
        *,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        task_type: object | None,
        bias: str,
        use_dora: bool,
    ) -> object: ...


class _GetPeftModel(Protocol):
    def __call__(self, model: nn.Module, peft_config: object) -> nn.Module: ...


class _PeftLoraModule(Protocol):
    TaskType: _TaskTypeEnum
    LoraConfig: _LoraConfigFactory
    get_peft_model: _GetPeftModel


class LoRALocus(Enum):
    """LoRA insertion locus sets."""

    ATTN_QV = "attn_qv"
    ATTN_ALL = "attn_all"
    FFN = "ffn"
    ATTN_QV_FFN = "attn_qv_ffn"
    EARLY_LAYERS = "early_layers"
    LATE_LAYERS = "late_layers"
    ALL = "all"


_T5_LOCUS_MODULES: dict[LoRALocus, list[str]] = {
    LoRALocus.ATTN_QV: ["q", "v"],
    LoRALocus.ATTN_ALL: ["q", "k", "v", "o"],
    LoRALocus.FFN: ["wi", "wo"],
    LoRALocus.ATTN_QV_FFN: ["q", "v", "wi", "wo"],
    LoRALocus.ALL: ["q", "k", "v", "o", "wi", "wo"],
}

_MOIRAI_LOCUS_MODULES: dict[LoRALocus, list[str]] = {
    LoRALocus.ATTN_QV: ["q_proj", "v_proj"],
    LoRALocus.ATTN_ALL: ["q_proj", "k_proj", "v_proj", "out_proj"],
    LoRALocus.FFN: ["fc1", "fc2"],
    LoRALocus.ATTN_QV_FFN: ["q_proj", "v_proj", "fc1", "fc2"],
    LoRALocus.ALL: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "fc_gate"],
}

_TIMESFM_LOCUS_MODULES: dict[LoRALocus, list[str]] = {
    LoRALocus.ATTN_QV: ["qkv_proj"],
    LoRALocus.ATTN_ALL: ["qkv_proj", "o_proj"],
    LoRALocus.FFN: ["gate_proj", "down_proj"],
    LoRALocus.ATTN_QV_FFN: ["qkv_proj", "gate_proj", "down_proj"],
    LoRALocus.ALL: ["qkv_proj", "o_proj", "gate_proj", "down_proj"],
}


@dataclass
class LoRAAdaptationConfig:
    """LoRA adaptation settings.

    Attributes:
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout probability.
        locus: Insertion locus set.
        target_modules: Explicit PEFT target modules, overrides locus.
        task_type: HuggingFace PEFT task type string.
        layers: Layer-depth filter in {"all", "early", "late"}.
        num_layers: Total model layer count for early/late filtering.
        layers_pattern: Pattern for extracting layer index from module name.
        architecture: Model architecture name for locus module mapping.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    locus: LoRALocus = LoRALocus.ATTN_ALL
    target_modules: list[str] | None = None
    task_type: str = "SEQ_2_SEQ_LM"
    layers: str = "all"
    num_layers: int = 12
    layers_pattern: str | None = "block"
    architecture: str = "t5"
    use_dora: bool = False


def _resolve_locus_modules(
    architecture: str,
    locus: LoRALocus,
) -> list[str]:
    """Resolve architecture-specific LoRA module suffixes from locus.

    Args:
        architecture: Model architecture identifier.
        locus: LoRA insertion locus set.

    Returns:
        Target module suffixes for PEFT.

    Raises:
        ValueError: If locus is unsupported for the architecture.
    """
    arch = architecture.lower()

    if "moirai" in arch:
        locus_modules = _MOIRAI_LOCUS_MODULES
    elif "timesfm" in arch:
        locus_modules = _TIMESFM_LOCUS_MODULES
    else:
        locus_modules = _T5_LOCUS_MODULES

    modules = locus_modules.get(locus)
    if modules is None:
        raise ValueError(f"Unknown LoRA locus for architecture={architecture}: {locus}")
    return modules


def _resolve_task_type(task_type_str: str) -> object | None:
    """Map string task type to PEFT TaskType.

    Args:
        task_type_str: TaskType string.

    Returns:
        TaskType instance or None.
    """

    peft_module = cast(_PeftLoraModule, cast(object, import_module("peft")))
    task_type_enum = peft_module.TaskType

    mapping = {
        "SEQ_2_SEQ_LM": task_type_enum.SEQ_2_SEQ_LM,
        "CAUSAL_LM": task_type_enum.CAUSAL_LM,
        "FEATURE_EXTRACTION": task_type_enum.FEATURE_EXTRACTION,
    }
    return mapping.get(task_type_str)


def _compute_layer_indices(
    layers_filter: str,
    num_layers: int,
) -> list[int] | None:
    """Compute layer indices for early/late layer filtering.

    Args:
        layers_filter: Layer filter in {"all", "early", "late"}.
        num_layers: Total layer count in the model.

    Returns:
        List of layer indices to transform, or None for all layers.
    """
    if layers_filter == "all":
        return None

    third = max(1, num_layers // 3)

    if layers_filter == "early":
        return list(range(0, third))
    elif layers_filter == "late":
        return list(range(num_layers - third, num_layers))
    else:
        logger.warning("Unknown layers filter: %s; using all layers", layers_filter)
        return None


def _resolve_explicit_targets(
    model: nn.Module,
    target_modules: list[str],
    layer_indices: list[int],
    layers_pattern: str,
) -> list[str]:
    """Resolve explicit target module names for layer-filtered LoRA.

    Instead of relying on PEFT's ``layers_to_transform`` regex (which
    fails when the ``layers_pattern`` appears at the start of a module
    path with no preceding dot), this function directly inspects the
    model's ``named_modules()`` to build an explicit list of full module
    paths to target.

    Args:
        model: The backbone model to inspect.
        target_modules: Base module name suffixes (e.g. ["q", "k", "v", "o"]).
        layer_indices: Which layer indices to include.
        layers_pattern: The sub-string that precedes the layer index in the
            module path (e.g. "block" for paths like ``block.3.layer.0.SelfAttention.q``).

    Returns:
        List of full dotted module names matching both the target suffix
        and the layer index filter.
    """
    import re

    # Build regex: (?:^|.*\.)block\.(\d+)\..*
    pattern = re.compile(rf"(?:^|.*\.){re.escape(layers_pattern)}\.(\d+)\..*")
    allowed_indices = set(layer_indices)
    explicit: list[str] = []

    for name, _ in model.named_modules():
        # Check if name ends with one of the target module suffixes
        short_name = name.rsplit(".", 1)[-1] if "." in name else name
        if short_name not in target_modules:
            continue

        # Extract layer index from full path
        match = pattern.match(name)
        if match is None:
            continue

        idx = int(match.group(1))
        if idx in allowed_indices:
            explicit.append(name)

    if not explicit:
        logger.warning(
            "No modules matched layer filter: pattern=%s, indices=%s, targets=%s",
            layers_pattern,
            sorted(allowed_indices),
            target_modules,
        )

    return explicit


def apply_lora(model: nn.Module, config: LoRAAdaptationConfig) -> nn.Module:
    """Apply LoRA adaptation to a model.

    For layer-level filtering (early/late layers), this function resolves
    explicit module name lists by inspecting the model's named modules.
    This avoids PEFT's ``layers_to_transform`` regex which can fail when
    the layer pattern appears at the start of a module path.

    Args:
        model: Original PyTorch model.
        config: LoRA adaptation config.

    Returns:
        PEFT model with LoRA adapters.
    """
    if config.target_modules is not None:
        target_modules: list[str] = config.target_modules
    else:
        target_modules = _resolve_locus_modules(
            architecture=config.architecture,
            locus=config.locus,
        )

    # Compute layer indices for early/late filtering
    layer_indices = _compute_layer_indices(
        layers_filter=config.layers,
        num_layers=config.num_layers,
    )

    # If layer filtering is needed, resolve explicit module paths
    if layer_indices is not None:
        layers_pattern = config.layers_pattern or "block"
        target_modules = _resolve_explicit_targets(
            model=model,
            target_modules=target_modules,
            layer_indices=layer_indices,
            layers_pattern=layers_pattern,
        )
        if not target_modules:
            raise ValueError(
                f"No target modules found for layers={config.layers}, "
                f"locus={config.locus.value}, num_layers={config.num_layers}"
            )

    task_type = _resolve_task_type(config.task_type)

    peft_module = cast(_PeftLoraModule, cast(object, import_module("peft")))
    lora_config_cls = peft_module.LoraConfig
    get_peft_model = peft_module.get_peft_model

    lora_config = lora_config_cls(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
        use_dora=config.use_dora,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in peft_model.parameters())
    ratio = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    method_name = "DoRA" if config.use_dora else "LoRA"
    logger.info(
        "%s applied: locus=%s, layers=%s, rank=%d, trainable=%s (%.2f%%)",
        method_name,
        config.locus.value,
        config.layers,
        config.rank,
        f"{trainable_params:,}",
        ratio,
    )

    return peft_model


def apply_dora(model: nn.Module, config: LoRAAdaptationConfig) -> nn.Module:
    """DoRA(Weight-Decomposed Low-Rank Adaptation)를 모델에 적용.

    use_dora=True로 설정하여 apply_lora를 호출하는 편의 함수.

    Args:
        model: 원본 PyTorch 모델.
        config: LoRA 적응 설정 (use_dora는 True로 강제 설정됨).

    Returns:
        DoRA 어댑터가 적용된 PEFT 모델.
    """
    import dataclasses

    dora_config = dataclasses.replace(config, use_dora=True)
    return apply_lora(model, dora_config)
