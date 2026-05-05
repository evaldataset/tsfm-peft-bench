from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Optional, Protocol, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DictConfig(Protocol):
    def __getattr__(self, name: str) -> object: ...


class _ChronosTokenizerProtocol(Protocol):
    def context_input_transform(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]: ...

    def label_input_transform(
        self,
        label: torch.Tensor,
        tokenizer_state: object,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def output_transform(
        self,
        samples: torch.Tensor,
        tokenizer_state: object,
    ) -> torch.Tensor: ...


class _ChronosInnerModelProtocol(Protocol):
    config: object
    model: nn.Module


class _ChronosPipelineInstanceProtocol(Protocol):
    tokenizer: _ChronosTokenizerProtocol
    model: _ChronosInnerModelProtocol

    def predict(self, *args: object, **kwargs: object) -> torch.Tensor: ...


class _ChronosPipelineClassProtocol(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> _ChronosPipelineInstanceProtocol: ...


class _ChronosModuleProtocol(Protocol):
    ChronosPipeline: _ChronosPipelineClassProtocol


class _T5ForConditionalGenerationClassProtocol(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> nn.Module: ...


class _TransformersModuleProtocol(Protocol):
    T5ForConditionalGeneration: _T5ForConditionalGenerationClassProtocol


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


@dataclass
class ChronosWrapperConfig:
    """Chronos 래퍼 설정.

    Args:
        hf_id: HuggingFace 모델 ID.
        context_length: 입력 컨텍스트 길이.
        prediction_length: 예측 길이.
        torch_dtype: 로딩 시 사용할 dtype 문자열.
        device_map: HuggingFace 로딩용 device_map.

    Returns:
        ChronosWrapperConfig 인스턴스.

    Raises:
        ValueError: ``context_length`` 또는 ``prediction_length``가 1 미만일 때.
    """

    hf_id: str
    context_length: int = 512
    prediction_length: int = 96
    torch_dtype: str | None = "bfloat16"
    device_map: str | None = None

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError(
                f"context_length는 1 이상이어야 합니다. 현재: {self.context_length}"
            )
        if self.prediction_length < 1:
            raise ValueError(
                f"prediction_length는 1 이상이어야 합니다. 현재: {self.prediction_length}"
            )

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> ChronosWrapperConfig:
        """Hydra DictConfig를 래퍼 설정으로 변환.

        Args:
            config: 모델 설정 DictConfig.

        Returns:
            ChronosWrapperConfig 인스턴스.

        Raises:
            ValueError: ``hf_id``가 없을 때.
        """

        hf_id = getattr(config, "hf_id", None)
        if not isinstance(hf_id, str) or hf_id == "":
            raise ValueError("config.hf_id는 비어 있지 않은 문자열이어야 합니다.")

        torch_dtype_obj = getattr(config, "torch_dtype", "bfloat16")
        if torch_dtype_obj is not None and not isinstance(torch_dtype_obj, str):
            raise ValueError("config.torch_dtype는 문자열 또는 None 이어야 합니다.")
        torch_dtype = cast(Optional[str], torch_dtype_obj)

        device_map_obj = getattr(config, "device_map", None)
        if device_map_obj is not None and not isinstance(device_map_obj, str):
            raise ValueError("config.device_map은 문자열 또는 None 이어야 합니다.")
        device_map = device_map_obj

        return cls(
            hf_id=hf_id,
            context_length=int(cast(int, getattr(config, "context_length", 512))),
            prediction_length=int(cast(int, getattr(config, "prediction_length", 96))),
            torch_dtype=torch_dtype,
            device_map=device_map,
        )


class ChronosWrapper:
    """Chronos T5 기반 모델 래퍼.

    Args:
        config: Hydra 모델 설정.

    Returns:
        ChronosWrapper 인스턴스.

    Raises:
        ValueError: 설정 값이 유효하지 않을 때.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config: ChronosWrapperConfig = ChronosWrapperConfig.from_dict_config(
            config
        )
        self.pipeline: _ChronosPipelineInstanceProtocol | None = None
        self.backbone: nn.Module | None = None
        self.tokenizer: _ChronosTokenizerProtocol | None = None
        self._native_prediction_length: int = self.config.prediction_length

    def _resolve_torch_dtype(self) -> torch.dtype | None:
        """문자열 dtype 설정을 torch dtype으로 변환.

        Args:
            None.

        Returns:
            torch.dtype 또는 None.

        Raises:
            ValueError: 지원하지 않는 dtype 문자열일 때.
        """

        if self.config.torch_dtype is None:
            return None

        dtype_name = self.config.torch_dtype
        dtype_obj = _DTYPE_MAP.get(dtype_name)
        if dtype_obj is None:
            raise ValueError(f"지원하지 않는 torch_dtype입니다: {dtype_name}")
        return dtype_obj

    def load(self) -> None:
        """HuggingFace에서 Chronos 파이프라인과 백본 로드.

        Args:
            None.

        Returns:
            None.

        Raises:
            ImportError: ``chronos`` 또는 ``transformers``를 찾을 수 없을 때.
            ValueError: 백본 추출에 실패했을 때.
        """

        dtype_obj = self._resolve_torch_dtype()
        load_kwargs: dict[str, object] = {}
        if dtype_obj is not None:
            load_kwargs["torch_dtype"] = dtype_obj

        if self.config.device_map is not None:
            load_kwargs["device_map"] = self.config.device_map
        else:
            load_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            chronos_module = cast(
                _ChronosModuleProtocol,
                cast(object, import_module("chronos")),
            )
        except ImportError as exc:
            raise ImportError(
                "chronos 패키지를 찾을 수 없습니다. `pip install chronos-forecasting`를 확인하세요."
            ) from exc

        pipeline_cls = chronos_module.ChronosPipeline
        self.pipeline = pipeline_cls.from_pretrained(self.config.hf_id, **load_kwargs)
        self.tokenizer = self.pipeline.tokenizer
        self._native_prediction_length: int = int(
            getattr(self.pipeline.model.config, "prediction_length", 64)
        )

        if hasattr(self.pipeline.model, "model"):
            self.backbone = self.pipeline.model.model
        else:
            try:
                transformers_module = cast(
                    _TransformersModuleProtocol,
                    cast(object, import_module("transformers")),
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers 패키지를 찾을 수 없습니다. `pip install transformers`를 확인하세요."
                ) from exc

            t5_cls = transformers_module.T5ForConditionalGeneration
            self.backbone = t5_cls.from_pretrained(self.config.hf_id, **load_kwargs)

        logger.info("Chronos 모델 로드 완료: %s", self.config.hf_id)

    def get_backbone(self) -> nn.Module:
        """PEFT 적용 대상 백본 모듈 반환.

        Args:
            None.

        Returns:
            T5 백본 모듈.

        Raises:
            ValueError: 모델이 아직 로드되지 않았을 때.
        """

        if self.backbone is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        return self.backbone

    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int = 20,
    ) -> torch.Tensor:
        """Chronos zero-shot 점 예측 수행.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            prediction_length: 예측 길이.
            num_samples: 샘플링 개수.

        Returns:
            중앙값 점예측 텐서, shape ``(batch, prediction_length)``.

        Raises:
            ValueError: 모델이 로드되지 않았거나 텐서 shape이 유효하지 않을 때.
        """

        if self.pipeline is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        if context.ndim == 1:
            context_batch = context.unsqueeze(0)
        elif context.ndim == 2:
            context_batch = context
        else:
            raise ValueError(
                f"context는 1D 또는 2D여야 합니다. 현재 shape: {context.shape}"
            )

        context_cpu = context_batch.detach().to(dtype=torch.float32, device="cpu")
        samples = self.pipeline.predict(
            inputs=context_cpu,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        if samples.ndim != 3:
            raise ValueError(
                f"Chronos 샘플 출력 shape이 예상과 다릅니다: {samples.shape}"
            )

        pred = samples.median(dim=1).values
        return pred.to(device=context.device, dtype=torch.float32)

    def __call__(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """forward() 로 위임하여 인스턴스 호출 지원.

        Args:
            context: 입력 컨텍스트.
            target: 예측 타깃.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.
        """

        return self.forward(context=context, target=target)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Chronos 학습용 forward 패스.

        Args:
            context: 입력 컨텍스트, shape ``(batch, context_len)`` 또는 ``(context_len,)``.
            target: 예측 타깃, shape ``(batch, prediction_len)`` 또는 ``(prediction_len,)``.

        Returns:
            ``{"loss": loss, "pred": pred}`` 딕셔너리.

        Raises:
            ValueError: 모델/토크나이저가 로드되지 않았거나 입력 shape이 유효하지 않을 때.
        """

        if self.backbone is None or self.tokenizer is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        if context.ndim == 1:
            context_batch = context.unsqueeze(0)
        elif context.ndim == 2:
            context_batch = context
        else:
            raise ValueError(
                f"context는 1D 또는 2D여야 합니다. 현재 shape: {context.shape}"
            )

        if target.ndim == 1:
            target_batch = target.unsqueeze(0)
        elif target.ndim == 2:
            target_batch = target
        else:
            raise ValueError(
                f"target은 1D 또는 2D여야 합니다. 현재 shape: {target.shape}"
            )

        if context_batch.shape[0] != target_batch.shape[0]:
            raise ValueError(
                f"배치 크기가 다릅니다: context={context_batch.shape[0]}, target={target_batch.shape[0]}"
            )

        model_device = next(self.backbone.parameters()).device
        # Chronos 토크나이저의 내부 bins가 CPU에 있으므로 CPU에서 토크나이징 수행
        context_for_tok = context_batch.detach().to(device="cpu", dtype=torch.float32)
        # Chronos 토크나이저는 native prediction_length만 허용
        native_pl = self._native_prediction_length
        target_for_tok = target_batch[:, :native_pl].detach().to(
            device="cpu", dtype=torch.float32
        )

        input_ids, attention_mask, tokenizer_state = (
            self.tokenizer.context_input_transform(context_for_tok)
        )
        labels, label_attention = self.tokenizer.label_input_transform(
            target_for_tok,
            tokenizer_state,
        )

        labels = labels.to(device=model_device)
        label_attention = label_attention.to(device=model_device)
        labels = labels.masked_fill(~label_attention.bool(), -100)

        outputs_obj = cast(
            object,
            self.backbone(
                input_ids=input_ids.to(device=model_device),
                attention_mask=attention_mask.to(device=model_device),
                labels=labels,
            ),
        )

        loss = cast(torch.Tensor, getattr(outputs_obj, "loss"))
        logits = cast(torch.Tensor, getattr(outputs_obj, "logits"))

        token_pred = torch.argmax(logits, dim=-1)
        horizon = min(int(target_batch.shape[-1]), native_pl)
        # output_transform 도 CPU에서 수행 (tokenizer bins가 CPU)
        token_pred_cpu = token_pred[:, :horizon].cpu()
        pred = self.tokenizer.output_transform(
            token_pred_cpu.unsqueeze(1),
            tokenizer_state,
        ).squeeze(1)

        pred = pred.to(device=target.device, dtype=target.dtype)
        return {"loss": loss, "pred": pred}

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """현재 학습 가능한 파라미터 목록 반환.

        Args:
            None.

        Returns:
            ``requires_grad=True`` 파라미터 리스트.

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        if self.backbone is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 load()를 호출하세요.")
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def eval(self) -> ChronosWrapper:
        """백본을 평가 모드로 전환.

        Args:
            None.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().eval()
        return self

    def train(self, mode: bool = True) -> ChronosWrapper:
        """백본을 학습 모드로 전환.

        Args:
            mode: True이면 학습 모드, False이면 평가 모드.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().train(mode)
        return self

    def to(self, device: torch.device | str) -> ChronosWrapper:
        """백본을 지정된 디바이스로 이동.

        Args:
            device: 대상 디바이스.

        Returns:
            self (메서드 체이닝 지원).

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        self.get_backbone().to(device)
        return self

    def parameters(self) -> object:
        """백본 파라미터 반환.

        Args:
            None.

        Returns:
            백본 모듈의 파라미터 이터레이터.

        Raises:
            ValueError: 모델이 로드되지 않았을 때.
        """

        return self.get_backbone().parameters()
