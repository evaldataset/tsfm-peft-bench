from __future__ import annotations

# pyright: reportMissingImports=false

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from src.adaptation.head import apply_head_only
from src.adaptation.lora import (
    LoRAAdaptationConfig,
    LoRALocus,
    _compute_layer_indices,
    apply_lora,
)
from src.adaptation.prefix import PrefixAdaptationConfig, apply_prefix_tuning


class _TinySelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _TinyDenseReluDense(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.wi = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)


class _TinyBlockLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.SelfAttention = _TinySelfAttention(dim)
        self.DenseReluDense = _TinyDenseReluDense(dim)


class _TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_TinyBlockLayer(dim)])


class TinyT5LikeModel(nn.Module):
    def __init__(self, dim: int = 4, num_blocks: int = 3) -> None:
        super().__init__()
        self.block = nn.ModuleList([_TinyBlock(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for block in self.block:
            layer = block.layer[0]
            y = layer.SelfAttention.o(layer.SelfAttention.v(y))
            y = layer.DenseReluDense.wo(layer.DenseReluDense.wi(y))
        return self.head(y)


class TestLoRALocusAndConfig:
    """LoRA 열거형과 설정 기본값을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_lora_locus_values(self) -> None:
        assert LoRALocus.ATTN_QV.value == "attn_qv"
        assert LoRALocus.ATTN_ALL.value == "attn_all"
        assert LoRALocus.FFN.value == "ffn"
        assert LoRALocus.ATTN_QV_FFN.value == "attn_qv_ffn"
        assert LoRALocus.EARLY_LAYERS.value == "early_layers"
        assert LoRALocus.LATE_LAYERS.value == "late_layers"
        assert LoRALocus.ALL.value == "all"

    def test_lora_config_defaults(self) -> None:
        cfg = LoRAAdaptationConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.05
        assert cfg.locus == LoRALocus.ATTN_ALL
        assert cfg.task_type == "SEQ_2_SEQ_LM"
        assert cfg.layers == "all"
        assert cfg.num_layers == 12


class TestLayerFiltering:
    """레이어 깊이 필터링 규칙을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_filter_all_returns_none(self) -> None:
        result = _compute_layer_indices(layers_filter="all", num_layers=12)
        assert result is None

    def test_filter_early_and_late(self) -> None:
        early = _compute_layer_indices(layers_filter="early", num_layers=12)
        late = _compute_layer_indices(layers_filter="late", num_layers=12)

        assert early is not None
        assert late is not None
        assert early == [0, 1, 2, 3]
        assert late == [8, 9, 10, 11]


class TestApplyLoRA:
    """apply_lora가 PEFT 팩토리를 올바르게 호출하는지 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_apply_lora_calls_peft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = TinyT5LikeModel()

        lora_config_mock = Mock(return_value={"kind": "lora-config"})
        get_peft_model_mock = Mock(return_value=model)

        task_type = SimpleNamespace(
            SEQ_2_SEQ_LM="seq2seq",
            CAUSAL_LM="causal",
            FEATURE_EXTRACTION="feat",
        )
        fake_peft = SimpleNamespace(
            TaskType=task_type,
            LoraConfig=lora_config_mock,
            get_peft_model=get_peft_model_mock,
        )

        monkeypatch.setattr("src.adaptation.lora.import_module", lambda _: fake_peft)

        cfg = LoRAAdaptationConfig(
            rank=4, alpha=8, dropout=0.1, locus=LoRALocus.ATTN_QV
        )
        result = apply_lora(model, cfg)
        assert result is model
        lora_config_mock.assert_called_once()
        kwargs = lora_config_mock.call_args.kwargs
        assert kwargs["r"] == 4
        assert kwargs["lora_alpha"] == 8
        assert kwargs["lora_dropout"] == pytest.approx(0.1)
        assert kwargs["target_modules"] == ["q", "v"]
        assert kwargs["task_type"] == "seq2seq"
        get_peft_model_mock.assert_called_once()


class TestApplyPrefixTuning:
    """apply_prefix_tuning의 PEFT 호출을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_apply_prefix_tuning_calls_peft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = TinyT5LikeModel()

        prefix_config_mock = Mock(return_value={"kind": "prefix-config"})
        get_peft_model_mock = Mock(return_value=model)
        task_type = SimpleNamespace(SEQ_2_SEQ_LM="seq2seq", CAUSAL_LM="causal")
        fake_peft = SimpleNamespace(
            TaskType=task_type,
            PrefixTuningConfig=prefix_config_mock,
            get_peft_model=get_peft_model_mock,
        )

        monkeypatch.setattr("src.adaptation.prefix.import_module", lambda _: fake_peft)

        cfg = PrefixAdaptationConfig(num_virtual_tokens=12, task_type="CAUSAL_LM")
        result = apply_prefix_tuning(model, cfg)
        assert result is model
        prefix_config_mock.assert_called_once_with(
            num_virtual_tokens=12,
            task_type="causal",
        )
        get_peft_model_mock.assert_called_once()


class TestApplyHeadOnly:
    """헤드 전용 미세조정 시 requires_grad 설정을 검증한다.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_only_head_params_unfrozen(self) -> None:
        model = TinyT5LikeModel()
        adapted = apply_head_only(model)

        for name, param in adapted.named_parameters():
            if "head" in name:
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False
